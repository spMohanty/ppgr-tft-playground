import os
import warnings
import random
import uuid
from dataclasses import dataclass, fields
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import click
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console

import json
import hashlib

import wandb
import matplotlib.pyplot as plt

from config import Config

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

# -----------------------------------------------------------------------------
# Data Loading and Preprocessing Functions
# -----------------------------------------------------------------------------
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
    
    # Create a custom progress layout
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        # Main user processing task
        user_task = progress.add_task("[cyan]Processing users...", total=len(df.groupby("user_id")))
        
        for user_id, group in df.groupby("user_id"):
            # Only allow food intake rows that have enough history and future for slicing
            food_intake_mask = group["food_intake_row"] == 1
            food_intake_mask.iloc[:max_encoder_length] = False
            food_intake_mask.iloc[-max_prediction_length:] = False
            
            food_intake_rows = group[food_intake_mask]
            user_slices = []
            
            # Nested progress for each user's food intake rows
            food_task = progress.add_task(
                f"[green]Processing User {user_id}", 
                total=len(food_intake_rows)
            )
            
            for row_idx, _ in food_intake_rows.iterrows():
                slice_start = row_idx - max_encoder_length + 1
                slice_end = row_idx + max_prediction_length + 1
                df_slice = df.iloc[slice_start:slice_end].copy()
                df_slice["time_series_cluster_id"] = str(uuid.uuid4())
                df_slice["time_idx"] = list(range(len(df_slice)))
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

# -----------------------------------------------------------------------------
# Time Series Dataset and Dataloader Creation
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# New: Cached Dataset Loader Function
# -----------------------------------------------------------------------------
def get_cached_time_series_datasets(config: Config, cache_dir: str = "cache/datasets") -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    os.makedirs(cache_dir, exist_ok=True)
    # Compute a hash based on all parameters that affect the dataset creation.
    params_dict = {
        "dataset_version": config.dataset_version,
        "debug_mode": config.debug_mode,
        "max_encoder_length": config.max_encoder_length,
        "max_prediction_length": config.max_prediction_length,
        "validation_percentage": config.validation_percentage,
        "test_percentage": config.test_percentage,
    }
    params_str = json.dumps(params_dict, sort_keys=True)
    cache_hash = hashlib.md5(params_str.encode("utf-8")).hexdigest()

    # Define cache file names
    train_cache_file = os.path.join(cache_dir, f"train_{cache_hash}.pt")
    val_cache_file = os.path.join(cache_dir, f"val_{cache_hash}.pt")
    test_cache_file = os.path.join(cache_dir, f"test_{cache_hash}.pt")

    if (not config.no_data_cache) and os.path.exists(train_cache_file) and os.path.exists(val_cache_file) and os.path.exists(test_cache_file):
        logger.warning(f"Loading cached datasets from {cache_dir} for hash {cache_hash}")
        training_dataset = torch.load(train_cache_file)
        validation_dataset = torch.load(val_cache_file)
        test_dataset = torch.load(test_cache_file)
    else:
        logger.info(f"Cache not found for hash {cache_hash}. Preparing datasets...")
        # Load the DataFrame
        df = load_dataframe(config.dataset_version, config.debug_mode)
        # Ensure the user_id is of type string
        df["user_id"] = df["user_id"].astype(str)

        # Prepare time series slices and split into train, val, test DataFrames
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
        
        # Create TimeSeriesDataSet objects using create_time_series_dataset
        training_dataset = create_time_series_dataset(
            train_df, config.max_encoder_length, config.max_prediction_length, food_covariates, predict_mode=False
        )
        validation_dataset = create_time_series_dataset(
            val_df, config.max_encoder_length, config.max_prediction_length, food_covariates, predict_mode=False
        )
        test_dataset = create_time_series_dataset(
            test_df, config.max_encoder_length, config.max_prediction_length, food_covariates, predict_mode=True
        )
        
        # Save the datasets to cache for future use
        torch.save(training_dataset, train_cache_file)
        torch.save(validation_dataset, val_cache_file)
        torch.save(test_dataset, test_cache_file)
        logger.info(f"Cached datasets saved to {cache_dir} with hash {cache_hash}")
    
    return training_dataset, validation_dataset, test_dataset


def create_dataloaders(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    test_dataset: TimeSeriesDataSet,
    batch_size: int,
    num_workers: int = 15,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders from TimeSeriesDataSet objects.
    """
    train_loader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=num_workers
    )
    val_loader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=num_workers
    )
    test_loader = test_dataset.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_cached_time_series_dataloaders(
    config: Config, cache_dir: str = "cache/datasets"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Retrieve cached TimeSeriesDataSets (or generate them if not cached),
    create DataLoaders, and cache them to disk.

    Note:
        Caching DataLoader objects using torch.save/torch.load can be tricky,
        especially with multi-worker setups (num_workers > 0). If you encounter
        pickling issues, consider setting num_workers=0 when caching or re-creating
        the DataLoaders on the fly.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation, and test loaders.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Compute a hash based on the parameters that affect dataset creation
    params_dict = {
        "dataset_version": config.dataset_version,
        "debug_mode": config.debug_mode,
        "max_encoder_length": config.max_encoder_length,
        "max_prediction_length": config.max_prediction_length,
        "validation_percentage": config.validation_percentage,
        "test_percentage": config.test_percentage,
    }
    params_str = json.dumps(params_dict, sort_keys=True)
    cache_hash = hashlib.md5(params_str.encode("utf-8")).hexdigest()
    
    # Define cache file names for the dataloaders
    train_loader_cache = os.path.join(cache_dir, f"train_loader_{cache_hash}.pt")
    val_loader_cache = os.path.join(cache_dir, f"val_loader_{cache_hash}.pt")
    test_loader_cache = os.path.join(cache_dir, f"test_loader_{cache_hash}.pt")
    
    
    if (not config.no_data_cache) and os.path.exists(train_loader_cache) and os.path.exists(val_loader_cache) and os.path.exists(test_loader_cache):
        logger.warning(f"Loading cached dataloaders from {cache_dir} for hash {cache_hash}")
        train_loader = torch.load(train_loader_cache)
        val_loader = torch.load(val_loader_cache)
        test_loader = torch.load(test_loader_cache)
    else:
        training_dataset, validation_dataset, test_dataset = get_cached_time_series_datasets(config, cache_dir)
        train_loader, val_loader, test_loader = create_dataloaders(
            training_dataset,
            validation_dataset,
            test_dataset,
            config.batch_size,
            num_workers=config.num_workers
        )
        torch.save(train_loader, train_loader_cache)
        torch.save(val_loader, val_loader_cache)
        torch.save(test_loader, test_loader_cache)
        logger.info(f"Cached dataloaders saved to {cache_dir} with hash {cache_hash}")
        
    return train_loader, val_loader, test_loader
