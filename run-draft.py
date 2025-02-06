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

from model import PPGRTemporalFusionTransformer

# -----------------------------------------------------------------------------
# Configuration Data Class
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # General configuration
    experiment_name: str = "tft-ppgr-2025-fo"
    
    random_seed: int = 42
    dataset_version: str = "v0.3"
    debug_mode: bool = False

    # WandB configuration
    wandb_project: str = "tft-ppgr-2025-debug"
    wandb_dir: str = "/scratch/mohanty/wandb"


    # Data slicing parameters
    max_encoder_length: int = 8 * 4  # encoder window length
    max_prediction_length: int = int(2 * 4)  # prediction horizon (2.5 hours)
    validation_percentage: float = 0.1
    test_percentage: float = 0.1

    # DataLoader parameters
    batch_size: int = 128
    num_workers: int = 4

    # Model hyperparameters
    learning_rate: float = 1e-2
    hidden_size: int = 256
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 256

    # Trainer parameters
    max_epochs: int = 200
    gradient_clip_val: float = 0.1

    # Early stopping parameters
    early_stop_monitor_metric: str = "val_loss"
    early_stop_monitor_metric_mode: str = "min"
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-4

    # Logger parameters
    
    # Checkpoint parameters
    checkpoint_monitor_metric: str = "val_loss"
    checkpoint_monitor_metric_mode: str = "min"
    checkpoint_dir: str = "/scratch/mohanty/checkpoints/tft-ppgr-2025"
    checkpoint_top_k: int = 5
    
    # Performance 
    
    disable_all_plots: bool = True
    

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
class PPGRMetricsCallback(pl.Callback):
    def __init__(self, mode: str = "val", num_plots: int = 6, disable_all_plots: bool = False):
        """
        Args:
            mode: either "val" or "test". This determines which hook methods are active
                  and the metric prefix used when logging.
            num_plots: number of random samples to plot
        """
        super().__init__()
        self.mode = mode
        self.num_plots = num_plots
        self.reset_metrics()
        self.correlation_calc_function = PearsonCorrCoef()
        self.final_metrics = {}
        self.plot_indices = None  # Will store the random indices
        self.metric_data = {}  # Store all metrics data
        self.disable_all_plots = disable_all_plots
        
    def reset_metrics(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_encoder_values = []
        
        self.raw_batch_data = []
        self.raw_prediction_outputs = []

    def _process_batch(self, pl_module, batch, outputs):        
        past_data, future_data = batch
        # outputs = pl_module(past_data) # Patching the TFT Class to return the output instead of recomputing
        predictions = outputs["prediction"]
        median_quantile_index = 3  # Using the median quantile for point forecasts
        point_forecast = predictions[:, :, median_quantile_index]
        self.all_predictions.append(point_forecast)
        self.all_targets.append(past_data["decoder_target"])
        self.all_encoder_values.append(past_data["encoder_target"])

        self.raw_batch_data.append(batch)
        self.raw_prediction_outputs.append(outputs)


    def plot_predictions(self, trainer, pl_module, batch_index_to_use: int = 0):
        if self.disable_all_plots: return
        
        batch_data = self.raw_batch_data[batch_index_to_use]
        past_data, future_data = batch_data
        prediction_outputs = self.raw_prediction_outputs[batch_index_to_use]
        
        # Initialize plot indices if not already set
        if self.plot_indices is None:
            batch_size = past_data["encoder_target"].shape[0]
            self.plot_indices = random.sample(range(batch_size), min(self.num_plots, batch_size))
            logger.info(f"Selected indices for plotting: {self.plot_indices}")
        
        # Plot each selected index
        for idx in self.plot_indices:
            _fig = pl_module.plot_prediction(past_data, prediction_outputs, idx)
            
            trainer.logger.experiment.log({
                f"predictions_sample_{idx}": wandb.Image(_fig)
            })

    # Activate the proper hook based on mode.
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.mode == "val":
            # TODO: refactor : I am sure theres a neater way to do this
            log, outputs = pl_module.validation_batch_full_outputs[-1] # The last one that has been added 
            self._process_batch(pl_module, batch, outputs)
            
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.mode == "test":
            # TODO: refactor : I am sure theres a neater way to do this
            log, outputs = pl_module.test_batch_full_outputs[-1] # The last one that has been added 
            self._process_batch(pl_module, batch, outputs)

    def plot_metric_scatter(self, metric_type: str, max_points: int = 10000):
        """
        Create scatter plots comparing predicted vs actual values for different metrics.
        
        Args:
            metric_type: String indicating which type of metric to plot ('iauc', 'auc', or 'cgm')
            max_points: Maximum number of points to plot (will randomly subsample if exceeded)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.disable_all_plots: return None
        
        if metric_type == 'iauc':
            metrics_to_plot = ["iauc_l2", "iauc_l3", "iauc_min_l2", "iauc_min_l3", "iauc_l1"]
            n_rows, n_cols = 2, 3
            figsize = (15, 10)
            title_prefix = 'iAUC'
        elif metric_type == 'auc':
            metrics_to_plot = ["auc"]
            n_rows, n_cols = 1, 1
            figsize = (6, 6)
            title_prefix = 'AUC'
        elif metric_type == 'cgm':
            metrics_to_plot = ["raw_cgm"]
            n_rows, n_cols = 1, 1
            figsize = (6, 6)
            title_prefix = 'CGM'
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        metrics_data = self.metric_data.get(metric_type)
        if metrics_data is None:
            logger.warning(f"No data available for metric type: {metric_type}")
            return None

        for idx, metric in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break
                
            pred = metrics_data["predictions"][metric].cpu().numpy()
            true = metrics_data["ground_truth"][metric].cpu().numpy()
            
            # Random subsampling if number of points exceeds max_points
            n_points = len(pred)
            if n_points > max_points:
                indices = np.random.choice(n_points, max_points, replace=False)
                pred = pred[indices]
                true = true[indices]
            
            # Create scatter plot
            sns.scatterplot(x=true, y=pred, s=0.75, alpha=0.5, ax=axes[idx])
                        
            # Add labels and title
            axes[idx].set_xlabel(f'True {title_prefix}')
            axes[idx].set_ylabel(f'Predicted {title_prefix}')
            axes[idx].set_title(f'{metric} Scatter Plot\n(n={len(pred):,} points)')
            
            # Use the pre-calculated correlation from self.final_metrics
            metric_key = f"{self.mode}_{metric}_corr"
            if metric_key in self.final_metrics:
                corr = self.final_metrics[metric_key].item()
                axes[idx].text(0.05, 0.95, f'r = {corr:.3f}', 
                             transform=axes[idx].transAxes, 
                             bbox=dict(facecolor='white', alpha=0.8))
        
        # Remove extra subplots if present
        if len(metrics_to_plot) < len(axes):
            for ax in axes[len(metrics_to_plot):]:
                fig.delaxes(ax)
            
        plt.tight_layout()
        
        # Add explicit figure close in case the caller doesn't use the returned figure
        plt.close()
        return fig

    def _compute_metrics(self):
        # Concatenate the lists of tensors along the batch dimension
        all_preds = torch.cat(self.all_predictions, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)
        all_encoder_values = torch.cat(self.all_encoder_values, dim=0)

        # Compute all metrics
        iauc_metrics = compute_iauc_metrics(all_targets, all_preds, all_encoder_values)
        auc_metrics = compute_auc_metrics(all_targets, all_preds, all_encoder_values)
        raw_cgm_metrics = {
            "predictions": {"raw_cgm": all_preds.flatten()},
            "ground_truth": {"raw_cgm": all_targets.flatten()}
        }
        
        # Store metrics data for plotting
        self.metric_data = {
            'iauc': iauc_metrics,
            'auc': auc_metrics,
            'cgm': raw_cgm_metrics
        }
        
        # Ensure correlation function is on the same device as the metrics
        self.correlation_calc_function = self.correlation_calc_function.to(all_targets.device)
        
        # Compute correlations
        iauc_correlations = compute_metric_correlations(iauc_metrics, self.correlation_calc_function)
        auc_correlations = compute_metric_correlations(auc_metrics, self.correlation_calc_function)
        raw_cgm_correlations = compute_metric_correlations(raw_cgm_metrics, self.correlation_calc_function)

        # Create and log scatter plots for each metric type
        if hasattr(self, 'trainer') and self.trainer is not None:
            for metric_type in ['iauc', 'auc', 'cgm']:
                scatter_fig = self.plot_metric_scatter(metric_type)
                if scatter_fig:
                    self.trainer.logger.experiment.log({
                        f"{self.mode}_{metric_type}_scatter": wandb.Image(scatter_fig)
                    })
                    plt.close(scatter_fig)

        # Combine the metrics into one dictionary with a prefix
        prefix = f"{self.mode}_"
        final = {}
        for metric_dict in [iauc_correlations, auc_correlations, raw_cgm_correlations]:
            for k, v in metric_dict.items():
                final[f"{prefix}{k}"] = v
        return final

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.mode == "val":
            self.trainer = trainer  # Store trainer reference for plotting
            final = self._compute_metrics()
            for key, value in final.items():
                pl_module.log(key, value, prog_bar=True, sync_dist=True)

            # Plot predictions
            self.plot_predictions(trainer, pl_module)

            self.final_metrics = final
            self.reset_metrics()
            

    def on_test_epoch_end(self, trainer, pl_module):
        if self.mode == "test":
            self.trainer = trainer  # Store trainer reference for plotting
            final = self._compute_metrics()
            for key, value in final.items():
                pl_module.log(key, value, prog_bar=True, sync_dist=True)
                
            # Plot predictions
            self.plot_predictions(trainer, pl_module)
                
            self.final_metrics = final
            self.reset_metrics()



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

    if os.path.exists(train_cache_file) and os.path.exists(val_cache_file) and os.path.exists(test_cache_file):
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
    
    if os.path.exists(train_loader_cache) and os.path.exists(val_loader_cache) and os.path.exists(test_loader_cache):
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


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def create_click_options(config_class):
    """Create click options automatically from a dataclass's fields."""
    def decorator(f):
        # Reverse the fields so the decorators are applied in the correct order
        for field in reversed(fields(config_class)):
            # Get the field type and default value
            param_type = field.type
            default_value = field.default
            
            # Convert type hints to click types
            type_mapping = {
                str: str,
                int: int,
                float: float,
                bool: bool
            }
            param_type = type_mapping.get(param_type, str)
            
            # Create the click option with both hyphen and underscore versions
            hyphen_name = f"--{field.name.replace('_', '-')}"
            underscore_name = f"--{field.name}"
            
            # Add aliases for specific parameters
            aliases = []
            if field.name == "debug_mode":
                aliases.append("--debug")
            
            # Combine all option names
            option_names = [hyphen_name, underscore_name] + aliases
            
            if param_type == bool:
                # Handle boolean flags differently
                f = click.option(
                    *option_names,
                    is_flag=True,
                    default=default_value,
                    help=f"Default: {default_value}"
                )(f)
            else:
                f = click.option(
                    *option_names,
                    type=param_type,
                    default=default_value,
                    help=f"Default: {default_value}"
                )(f)
        return f
    return decorator

@click.command()
@create_click_options(Config)
def main(**kwargs):
    """
    All CLI arguments are automatically collected in kwargs
    and passed to Config constructor
    """
    
    # Initialize configuration with CLI parameters
    config = Config(**kwargs)
    
    # Create a meaningful experiment name based on hyperparameters
    base_name = config.experiment_name
    experiment_name_parts = [
        base_name,  # Keep base name as prefix
        f"hs{config.hidden_size}",
        f"ahs{config.attention_head_size}",
        f"hcs{config.hidden_continuous_size}",
        f"d{int(config.dropout*100)}",
        f"lr{config.learning_rate:.0e}",
        f"bs{config.batch_size}"
    ]
    config.experiment_name = "-".join(experiment_name_parts)
    
    # Set wandb directory before initializing
    os.makedirs(config.wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = config.wandb_dir
    print(f"WANDB_DIR: {os.environ['WANDB_DIR']}")

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.experiment_name,
        dir=config.wandb_dir,
        settings=wandb.Settings(start_method="thread")
    )
    print(f"Wandb directory: {wandb.run.dir}")


    # Initialize wandb first
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.experiment_name
    )
    
    # Debug overrides
    if config.debug_mode:
        config.max_epochs = 1
        
    # When running a sweep, wandb.config will contain the current run's hyperparameters.
    # Override the default config values if they exist in wandb.config.
    if hasattr(wandb, 'config') and wandb.run is not None:
        config.learning_rate = wandb.config.get("learning_rate", config.learning_rate)
        config.hidden_size = wandb.config.get("hidden_size", config.hidden_size)
        config.attention_head_size = wandb.config.get("attention_head_size", config.attention_head_size)
        config.dropout = wandb.config.get("dropout", config.dropout)
        config.hidden_continuous_size = wandb.config.get("hidden_continuous_size", config.hidden_continuous_size)
        config.max_epochs = wandb.config.get("max_epochs", config.max_epochs)
        config.batch_size = wandb.config.get("batch_size", config.batch_size)
        
        # Update experiment name with sweep parameters
        experiment_name_parts = [
            base_name,  # Keep base name as prefix
            f"hs{config.hidden_size}",
            f"ahs{config.attention_head_size}",
            f"hcs{config.hidden_continuous_size}",
            f"d{int(config.dropout*100)}",
            f"lr{config.learning_rate:.0e}",
            f"bs{config.batch_size}"
        ]
        new_name = "-".join(experiment_name_parts)
        wandb.run.name = new_name
        wandb.run.save()

    # Create hyperparameter dictionary for wandb logging
    hyperparameters = {
        # General configuration
        "random_seed": config.random_seed,
        "dataset_version": config.dataset_version,
        "debug_mode": config.debug_mode,
        
        # Data slicing parameters
        "max_encoder_length": config.max_encoder_length,
        "max_prediction_length": config.max_prediction_length,
        "validation_percentage": config.validation_percentage,
        "test_percentage": config.test_percentage,
        
        # DataLoader parameters
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        
        # Model hyperparameters
        "learning_rate": config.learning_rate,
        "hidden_size": config.hidden_size,
        "attention_head_size": config.attention_head_size,
        "dropout": config.dropout,
        "hidden_continuous_size": config.hidden_continuous_size,
        
        # Trainer parameters
        "max_epochs": config.max_epochs,
        "gradient_clip_val": config.gradient_clip_val,
        
        # Early stopping parameters
        "early_stop_patience": config.early_stop_patience,
        "early_stop_min_delta": config.early_stop_min_delta,
    }

    # Update wandb config with all hyperparameters
    wandb_logger.log_hyperparams(hyperparameters)

    # Set random seed and other configurations
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    train_loader, val_loader, test_loader = get_cached_time_series_dataloaders(config)

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        logger.info("CUDA is not available. Using CPU.")

    # Build the Temporal Fusion Transformer model using the config parameters
    tft_model = PPGRTemporalFusionTransformer.from_dataset(
        train_loader.dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        loss=QuantileLoss(),
        log_interval=5,  # Adjust as needed for logging frequency
        optimizer="ranger",
        reduce_on_plateau_patience=4,
    )
    logger.info(f"Number of parameters in network: {tft_model.size() / 1e3:.1f}k")

    # Build the PyTorch Lightning callbacks
    early_stop_callback = EarlyStopping(
        monitor=config.early_stop_monitor_metric,
        min_delta=config.early_stop_min_delta,
        patience=config.early_stop_patience,
        verbose=False,
        mode=config.early_stop_monitor_metric_mode
    )
    lr_logger = LearningRateMonitor()  # Logs the learning rate during training

    # Instantiate the custom iAUC callback with the validation dataloader and compute function
    ppgr_metrics_val_callback = PPGRMetricsCallback(mode="val", disable_all_plots=config.disable_all_plots)
    ppgr_metrics_test_callback = PPGRMetricsCallback(mode="test", disable_all_plots=config.disable_all_plots)

    # Instantiate the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=config.checkpoint_monitor_metric,
        mode=config.checkpoint_monitor_metric_mode,
        save_top_k=config.checkpoint_top_k,  # Save only the best model.
        dirpath=config.checkpoint_dir,
        filename="{epoch}-{val_loss:.4f}" 
    )

    # Build the PyTorch Lightning trainer with all callbacks
    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=config.max_epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[lr_logger, early_stop_callback, ppgr_metrics_val_callback, ppgr_metrics_test_callback],
        logger=wandb_logger,
        val_check_interval=0.25,  # Run validation 3 times per epoch
        check_val_every_n_epoch=1  # Ensure we check validation every epoch
    )

    # Start training
    trainer.fit(tft_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Run test loop
    trainer.test(tft_model, dataloaders=test_loader)

    # Extract final test metrics from the test callback.
    final_test_metrics = ppgr_metrics_test_callback.final_metrics
    print("Final Test Metrics:")
    for metric_name, metric_value in final_test_metrics.items():
        # Assuming metric_value is a tensor, convert to a Python float.
        print(metric_name, metric_value.item())


if __name__ == "__main__":
    main()
