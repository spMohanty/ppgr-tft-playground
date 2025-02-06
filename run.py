import os
import warnings
import random
import uuid
from dataclasses import dataclass
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

# PyTorch Forecasting and Lightning imports
from pytorch_forecasting import (
    Baseline,
    TemporalFusionTransformer,
    TimeSeriesDataSet
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from torchmetrics.regression import PearsonCorrCoef

# -----------------------------------------------------------------------------
# Configuration Data Class
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # General configuration
    random_seed: int = 42
    dataset_version: str = "v0.3"
    debug_mode: bool = False

    # Data slicing parameters
    max_encoder_length: int = 12 * 4  # encoder window length
    max_prediction_length: int = int(2.5 * 4)  # prediction horizon (2.5 hours)
    validation_percentage: float = 0.1
    test_percentage: float = 0.1

    # DataLoader parameters
    batch_size: int = 64
    num_workers: int = 15

    # Model hyperparameters
    learning_rate: float = 0.03
    hidden_size: int = 8
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 8

    # Trainer parameters
    max_epochs: int = 200
    gradient_clip_val: float = 0.1

    # Early stopping parameters
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-4

    # Logger parameters
    wandb_project: str = "tft-ppgr-2025"

# -----------------------------------------------------------------------------
# Custom iAUC Metric
# -----------------------------------------------------------------------------
# Compute area using trapezoidal rule
# torch.trapz needs values to be in float32/64    
def _compute_iauc(predicted, baseline):
    incremental = torch.maximum(predicted - baseline, torch.tensor(0.0))
    return torch.trapz(incremental.float(), dim=1)

def compute_iauc_metrics(targets: torch.Tensor, predictions: torch.Tensor, encoder_values: torch.Tensor) -> float:
    """
    Compute the incremental Area Under the Curve (iAUC) for each sample using PyTorch operations.
    
    Args:
        targets: Tensor of shape (batch_size, prediction_length)
        predictions: Tensor of shape (batch_size, prediction_length)
        encoder_values: Tensor of shape (batch_size, encoder_length)
            containing the context window values
    
    Returns:
        float: Mean iAUC across all samples
    """
    # Calculate baseline as mean of last 2 values from context window
    baseline_mean_l2 = encoder_values[:, -2:].mean(dim=1, keepdim=True)  # (batch_size, 1)
    baseline_mean_l3 = encoder_values[:, -3:].mean(dim=1, keepdim=True)  # (batch_size, 1)
    baseline_min_l2 = encoder_values[:, -2:].min(dim=1, keepdim=True).values  # (batch_size, 1)
    baseline_min_l3 = encoder_values[:, -3:].min(dim=1, keepdim=True).values  # (batch_size, 1)
    baseline_l1 = encoder_values[:, -1:].mean(dim=1, keepdim=True)  # (batch_size, 1)
    
    # Compute iAUC for predictions for each baseline
    iauc_l2_pred = _compute_iauc(predictions, baseline_mean_l2)
    iauc_l3_pred = _compute_iauc(predictions, baseline_mean_l3)
    iauc_min_l2_pred = _compute_iauc(predictions, baseline_min_l2)
    iauc_min_l3_pred = _compute_iauc(predictions, baseline_min_l3)
    iauc_l1_pred = _compute_iauc(predictions, baseline_l1)

    # Compute iAUC for targets for each baseline
    iauc_l2_ground_truth = _compute_iauc(targets, baseline_mean_l2)
    iauc_l3_ground_truth = _compute_iauc(targets, baseline_mean_l3)
    iauc_min_l2_ground_truth = _compute_iauc(targets, baseline_min_l2)
    iauc_min_l3_ground_truth = _compute_iauc(targets, baseline_min_l3)
    iauc_l1_ground_truth = _compute_iauc(targets, baseline_l1)
        
    return {
        "predictions": {
            "iauc_l2": iauc_l2_pred,
            "iauc_l3": iauc_l3_pred,
            "iauc_min_l2": iauc_min_l2_pred,
            "iauc_min_l3": iauc_min_l3_pred,
            "iauc_l1": iauc_l1_pred
        },
        "ground_truth": {
            "iauc_l2": iauc_l2_ground_truth,
            "iauc_l3": iauc_l3_ground_truth,
            "iauc_min_l2": iauc_min_l2_ground_truth,
            "iauc_min_l3": iauc_min_l3_ground_truth,
            "iauc_l1": iauc_l1_ground_truth
        }
    }

def compute_auc_metrics(targets: torch.Tensor, predictions: torch.Tensor, encoder_values: torch.Tensor) -> float:
    """
    Compute the Area Under the Curve (AUC) for each sample using PyTorch operations.
    
    Args:
        targets: Tensor of shape (batch_size, prediction_length)
        predictions: Tensor of shape (batch_size, prediction_length)
        encoder_values: Tensor of shape (batch_size, encoder_length)
            containing the context window values
    
    Returns:
        float: Mean AUC across all samples
    """
    
    # Calculate AUC for predictions  (Note: computing iauc with baseline=0 is equivalent to computing auc)
    auc_pred = _compute_iauc(predictions, baseline=0)
    auc_ground_truth = _compute_iauc(targets, baseline=0)
    
    return {
        "predictions": {
            "auc": auc_pred
        },
        "ground_truth": {
            "auc": auc_ground_truth
        }
    }
    
    
def compute_metric_correlations(metrics: dict, correlation_calc_fn: Any) -> dict:
    """
    Compute the correlation between the predictions and the ground truth for each metric.
    """
    # assumes a dictionary with "predictions" and "ground_truth" keys
    # with each metric being present in both dictionaries
    # The values are torch tensors matching the corresponding shapes
    # returns a dictionary with the correlation between the predictions and the ground truth for each metric
    correlations = {}
    for metric_name, _ in metrics["predictions"].items():
        predicted_values = metrics["predictions"][metric_name]
        ground_truth_values = metrics["ground_truth"][metric_name]
            
        # Calculate the coorelation using the provided function (PearsonCorrCoef)        
        correlations[f"{metric_name}_corr"] = correlation_calc_fn(predicted_values, ground_truth_values)
        
        
    return correlations


# -----------------------------------------------------------------------------
# Custom Callback for iAUC Validation Metric
# -----------------------------------------------------------------------------
class IAUCValidationCallback(pl.Callback):
    def __init__(self):
        """
        """
        super().__init__()
        self.reset_validation_metrics()
        
        self.correlation_calc_function = PearsonCorrCoef()

    def reset_validation_metrics(self):
        """Reset accumulated metrics at the start of validation."""
        self.all_predictions = []
        self.all_targets = []
        self.all_encoder_values = []

    def on_validation_start(self, trainer, pl_module):
        """Reset metrics at the start of validation."""
        self.reset_validation_metrics()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Accumulate predictions and targets from each batch."""
                
        # Unpack TFT's temporal components from batch
        past_data, future_data = batch
        
        # Do an explicity forward pass to get the predictions
        # TODO: check why outputs is not carrying this information
        _outputs = pl_module(past_data)
        
        predictions = _outputs["prediction"]
        median_quantile_index = 3  # 3 is the median quantile index
        point_forecast = predictions[:, :, median_quantile_index]
        
        
        # gather all context
        self.all_encoder_values.append(past_data["encoder_target"])
        self.all_targets.append(past_data["decoder_target"])
        self.all_predictions.append(point_forecast)
        
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Calculate iAUC using accumulated predictions and targets."""
        # Concatenate all accumulated tensors
        all_preds = torch.cat(self.all_predictions, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)
        all_encoder_values = torch.cat(self.all_encoder_values, dim=0)

        # Compute iAUC using the provided function
        iauc_metrics = compute_iauc_metrics(all_targets, all_preds, all_encoder_values)
        auc_metrics = compute_auc_metrics(all_targets, all_preds, all_encoder_values)
        
        # Ensure that correlation calc function is on the same device as the metrics
        self.correlation_calc_function.to(all_preds.device)
        
        
        
        
        # Compute iAUC correlations
        iauc_correlations = compute_metric_correlations(iauc_metrics, self.correlation_calc_function)
        
        # Compute AUC correlations
        auc_correlations = compute_metric_correlations(auc_metrics, self.correlation_calc_function)
        
        # Compute raw CGM correlations
        raw_cgm_metrics = {
            "predictions": {
                "raw_cgm": all_preds.flatten(),
            },
            "ground_truth": {
                "raw_cgm": all_targets.flatten()
            }
        }
        raw_cgm_correlations = compute_metric_correlations(raw_cgm_metrics, self.correlation_calc_function)
        
        
        
        # Log the custom metrics
        ## iAUC 
        for metric_name, metric_value in iauc_correlations.items():
            pl_module.log(f"val_{metric_name}", metric_value, prog_bar=True)
        
        # AUC
        for metric_name, metric_value in auc_correlations.items():
            pl_module.log(f"val_{metric_name}", metric_value, prog_bar=True)
        
        # RAW CGM values
        for metric_name, metric_value in raw_cgm_correlations.items():
            pl_module.log(f"val_{metric_name}", metric_value, prog_bar=True)
        
        # Clean up
        self.reset_validation_metrics()

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


def create_dataloaders(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
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
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
@click.command()
@click.option('--debug', is_flag=True, default=False, help='Enable debug mode')
def main(debug):
    # Initialize configuration (override the debug flag from the command-line)
    config = Config(debug_mode=debug)
    
    # Debug overrides
    if config.debug_mode:
        config.max_epochs = 10
    
    # Set random seed and other configurations
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # Load the data
    df = load_dataframe(config.dataset_version, config.debug_mode)
    df["user_id"] = df["user_id"].astype(str)

    # Prepare time series slices and split into train/val/test DataFrames
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

    # Determine food covariates (all columns starting with "food__")
    food_covariates = [col for col in train_df.columns if col.startswith("food__")]

    # Create TimeSeriesDataSet objects for training and validation
    training_dataset = create_time_series_dataset(
        train_df, config.max_encoder_length, config.max_prediction_length, food_covariates, predict_mode=False
    )
    validation_dataset = create_time_series_dataset(
        val_df, config.max_encoder_length, config.max_prediction_length, food_covariates, predict_mode=False
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        training_dataset, validation_dataset, config.batch_size, num_workers=config.num_workers
    )

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        logger.info("CUDA is not available. Using CPU.")

    # Build the Temporal Fusion Transformer model using the config parameters
    tft_model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        loss=QuantileLoss(),
        log_interval=50,  # Adjust as needed for logging frequency
        optimizer="ranger",
        reduce_on_plateau_patience=4,
    )
    logger.info(f"Number of parameters in network: {tft_model.size() / 1e3:.1f}k")

    # Build the PyTorch Lightning callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=config.early_stop_min_delta,
        patience=config.early_stop_patience,
        verbose=False,
        mode="min"
    )
    lr_logger = LearningRateMonitor()  # Logs the learning rate during training

    # Instantiate the custom iAUC callback with the validation dataloader and compute function
    iauc_callback = IAUCValidationCallback()

    # Set up the WandB logger
    wandb_logger = WandbLogger(project=config.wandb_project)

    # Build the PyTorch Lightning trainer with all callbacks
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[lr_logger, early_stop_callback, iauc_callback],
        logger=wandb_logger,
    )

    # Start training
    trainer.fit(tft_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
