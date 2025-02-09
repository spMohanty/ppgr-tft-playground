import os
import json
import uuid
import hashlib
import random
import warnings
from dataclasses import dataclass, fields
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import click
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console

import wandb
import matplotlib.pyplot as plt

# PyTorch Forecasting and Lightning imports
from pytorch_forecasting import (
    Baseline,
    TemporalFusionTransformer,
    TimeSeriesDataSet
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from torchmetrics.regression import PearsonCorrCoef

from config import Config

# =============================================================================
# Helper functions for caching
# =============================================================================
def get_cache_params(config: Any) -> dict:
    """Extract the parameters from the config that affect dataset/dataloader creation."""
    return {
        "dataset_version": config.dataset_version,
        "debug_mode": config.debug_mode,
        "max_encoder_length": config.max_encoder_length,
        "max_prediction_length": config.max_prediction_length,
        "validation_percentage": config.validation_percentage,
        "test_percentage": config.test_percentage,
    }


def compute_cache_hash(params: dict) -> str:
    """Compute a hash string from a dictionary of parameters."""
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode("utf-8")).hexdigest()


def get_cache_file_paths(
    config: Any, cache_dir: str, prefixes: List[str]
) -> Dict[str, str]:
    """
    Given a list of filename prefixes, return a mapping of prefix to a cache file path.
    For example, prefixes might be ["train", "val", "test"].
    """
    params = get_cache_params(config)
    cache_hash = compute_cache_hash(params)
    paths = {prefix: os.path.join(cache_dir, f"{prefix}_{cache_hash}.pt") for prefix in prefixes}
    return paths


# =============================================================================
# Data Loading and Preprocessing Functions
# =============================================================================
def load_dataframe(dataset_version: str, debug_mode: bool) -> pd.DataFrame:
    """
    Load the processed CSV file into a DataFrame and sort by user, block, and time.
    """
    if debug_mode:
        file_path = (
            f"data/processed/{dataset_version}/debug/"
            f"debug-fay-ppgr-processed-and-aggregated-{dataset_version}.csv"
        )
    else:
        file_path = (
            f"data/processed/{dataset_version}/"
            f"fay-ppgr-processed-and-aggregated-{dataset_version}.csv"
        )
    df = pd.read_csv(file_path)
    df = df.sort_values(by=["user_id", "timeseries_block_id", "read_at"])
    logger.info(f"Loaded dataframe with {len(df)} rows from {file_path}")
    return df


def create_slice(
    df: pd.DataFrame, slice_start: int, slice_end: int
) -> pd.DataFrame:
    """Helper to create a time-series slice with a unique cluster id and time index."""
    df_slice = df.iloc[slice_start:slice_end].copy()
    df_slice["time_series_cluster_id"] = str(uuid.uuid4())
    df_slice["time_idx"] = list(range(len(df_slice)))
    return df_slice


def prepare_time_series_slices(
    df: pd.DataFrame,
    max_encoder_length: int,
    max_prediction_length: int,
    validation_percentage: float,
    test_percentage: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create time-series slices anchored around food intake rows and split them
    into training, validation, and test sets.
    """
    training_data_slices = []
    validation_data_slices = []
    test_data_slices = []
    
    console = Console()
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )
    
    with progress:
        user_task = progress.add_task("[cyan]Processing users...", total=len(df.groupby("user_id")))
        
        for user_id, group in df.groupby("user_id"):
            # Only allow food intake rows that have enough history and future for slicing
            food_intake_mask = group["food_intake_row"] == 1
            # Ensure that there is enough past and future data
            food_intake_mask.iloc[:max_encoder_length] = False
            food_intake_mask.iloc[-max_prediction_length:] = False
            food_intake_rows = group[food_intake_mask]
            
            user_slices = []
            food_task = progress.add_task(f"[green]Processing User {user_id}", total=len(food_intake_rows))
            
            for row_idx, _ in food_intake_rows.iterrows():
                slice_start = row_idx - max_encoder_length + 1
                slice_end = row_idx + max_prediction_length + 1
                df_slice = create_slice(df, slice_start, slice_end)
                user_slices.append(df_slice)
                progress.update(food_task, advance=1)
            
            # Split the slices for the user
            n_samples = len(user_slices)
            if n_samples > 0:
                train_end = int((1 - validation_percentage - test_percentage) * n_samples)
                val_end = int((1 - test_percentage) * n_samples)
                training_data_slices.extend(user_slices[:train_end])
                validation_data_slices.extend(user_slices[train_end:val_end])
                test_data_slices.extend(user_slices[val_end:])
            
            progress.update(user_task, advance=1)
            progress.remove_task(food_task)
    
    training_df = pd.concat(training_data_slices)
    validation_df = pd.concat(validation_data_slices)
    test_df = pd.concat(test_data_slices)
    return training_df, validation_df, test_df


def create_time_series_dataset(
    df: pd.DataFrame,
    max_encoder_length: int,
    max_prediction_length: int,
    food_covariates: List[str],
    predict_mode: bool = False,
) -> TimeSeriesDataSet:
    """
    Create a TimeSeriesDataSet from a DataFrame for PyTorch Forecasting.
    """
    df = df.reset_index(drop=True)
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="val",
        group_ids=["user_id", "time_series_cluster_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["user_id"],
        time_varying_known_categoricals=[
            "loc_eaten_dow",
            "loc_eaten_dow_type",
            "loc_eaten_season"
        ],
        time_varying_known_reals=["time_idx", "loc_eaten_hour"] + food_covariates,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["val"],
        target_normalizer=GroupNormalizer(
            groups=["user_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        predict_mode=predict_mode,
    )
    return dataset


# =============================================================================
# Cached Dataset and DataLoader Functions
# =============================================================================
def get_cached_time_series_datasets(
    config: Any
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Retrieve cached TimeSeriesDataSets (or generate them if not cached)
    for training, validation, and test.
    """
    cache_dir = config.dataset_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    params = get_cache_params(config)
    cache_hash = compute_cache_hash(params)
    paths = get_cache_file_paths(config, cache_dir, ["train", "val", "test"])
    
    if (not config.no_data_cache) and all(os.path.exists(p) for p in paths.values()):
        logger.warning(f"Loading cached datasets from {cache_dir} for hash {cache_hash}")
        training_dataset = torch.load(paths["train"], weights_only=False)
        validation_dataset = torch.load(paths["val"], weights_only=False)
        test_dataset = torch.load(paths["test"], weights_only=False)
    else:
        logger.info(f"Cache not found for hash {cache_hash}. Preparing datasets...")
        df = load_dataframe(config.dataset_version, config.debug_mode)
        df["user_id"] = df["user_id"].astype(str)
        train_df, val_df, test_df = prepare_time_series_slices(
            df,
            config.max_encoder_length,
            config.max_prediction_length,
            config.validation_percentage,
            config.test_percentage
        )
        logger.info(
            f"Time series slices created - Train: {len(train_df)} rows, "
            f"Val: {len(val_df)} rows, Test: {len(test_df)} rows"
        )
        
        # Determine food covariates (all columns beginning with "food__")
        food_covariates = [col for col in train_df.columns if col.startswith("food__")]
        
        training_dataset = create_time_series_dataset(
            train_df, config.max_encoder_length, config.max_prediction_length, food_covariates, predict_mode=False
        )
        validation_dataset = create_time_series_dataset(
            val_df, config.max_encoder_length, config.max_prediction_length, food_covariates, predict_mode=False
        )
        test_dataset = create_time_series_dataset(
            test_df, config.max_encoder_length, config.max_prediction_length, food_covariates, predict_mode=True
        )
        
        torch.save(training_dataset, paths["train"])
        torch.save(validation_dataset, paths["val"])
        torch.save(test_dataset, paths["test"])
        logger.info(f"Cached datasets saved to {cache_dir} with hash {cache_hash}")
    
    return training_dataset, validation_dataset, test_dataset


def create_dataloaders(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    test_dataset: TimeSeriesDataSet,
    batch_size: int,
    num_workers: int = 15,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    shuffle_test: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects from TimeSeriesDataSet objects.
    """
    pin_memory = torch.cuda.is_available()
    train_loader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_train, pin_memory=pin_memory, persistent_workers=True
    )
    val_loader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=num_workers, shuffle=shuffle_val, pin_memory=pin_memory, persistent_workers=True
    )
    test_loader = test_dataset.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=num_workers, shuffle=shuffle_test, pin_memory=pin_memory, persistent_workers=True 
    )
    
    return train_loader, val_loader, test_loader


def get_time_series_dataloaders(
    config: Config,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    shuffle_test: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Retrieve the cached TimeSeriesDataSet objects (or generate them if not cached),
    and then create DataLoader objects from these datasets.

    Caching is done only for the dataset objects, not for the dataloaders.
    """
    # Retrieve (or generate) the cached datasets.
    training_dataset, validation_dataset, test_dataset = get_cached_time_series_datasets(config)
    
    # Create DataLoaders from the cached datasets.
    train_loader, val_loader, test_loader = create_dataloaders(
        training_dataset,
        validation_dataset,
        test_dataset,
        config.batch_size,
        num_workers=config.num_workers,
        shuffle_train=shuffle_train,
        shuffle_val=shuffle_val,
        shuffle_test=shuffle_test,
    )
    
    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    from utils import set_random_seeds
    
    from rich import print
    
    # Set seeds for reproducibility.
    set_random_seeds(42)
    
    # Create a Config object. 
    # check config.py for details
    config = Config()
    
    config.debug_mode = True

    print("=== Testing Cached Time Series DataLoaders ===")
    
    try:
        train_loader, val_loader, test_loader = get_time_series_dataloaders(config)
    except Exception as e:
        print("An error occurred while creating dataloaders:", e)
        exit(1)
    
    # Print basic info about each loader.
    print(f"\nTrain loader has {len(train_loader)} batches.")
    print(f"Validation loader has {len(val_loader)} batches.")
    print(f"Test loader has {len(test_loader)} batches.")

    # # Print a sample batch from the train_loader.
    print("\n=== Sample Batch from Train Loader ===")
    for batch in train_loader:
        past_data, future_data = batch
        print(f"past_data: {past_data}")
        print(f"future_data: {future_data}")
                
        break  # Only process one batch for testing purposes.
    
    print("\nTest run completed successfully!")



    
