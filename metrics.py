import random
import numpy as np
import torch
from loguru import logger
import wandb
import matplotlib.pyplot as plt

from typing import List, Tuple, Any

import lightning.pytorch as pl
from torchmetrics.regression import PearsonCorrCoef


# -----------------------------------------------------------------------------
# Custom iAUC Metric
# -----------------------------------------------------------------------------
# Compute area using trapezoidal rule
def _compute_iauc(predicted, baseline, allow_negative: bool = False):
    # compute the incremental area under the curve
    incremental = predicted - baseline
    
    # Clip negative values if not allowed
    if not allow_negative:
        incremental = torch.maximum(incremental, torch.tensor(0.0))
        
    return torch.trapz(incremental.float(), dim=1)

def compute_iauc_metrics(
    targets: torch.Tensor, 
    predictions: torch.Tensor, 
    encoder_values: torch.Tensor
) -> dict:
    """
    Compute the incremental Area Under the Curve (iAUC) for each sample using PyTorch tensors.
    
    Args:
        targets: Tensor of shape (batch_size, prediction_length)
        predictions: Tensor of shape (batch_size, prediction_length)
        encoder_values: Tensor of shape (batch_size, encoder_length)
            containing the context window values

    Returns:
        dict: A dictionary with iAUC metrics for predictions and ground truth.
    """
    # Compute baselines using different slices/aggregations of the encoder_values.
    baselines = {
        "l2": encoder_values[:, -2:].mean(dim=1, keepdim=True),
        "l3": encoder_values[:, -3:].mean(dim=1, keepdim=True),
        "min_l2": encoder_values[:, -2:].min(dim=1, keepdim=True).values,
        "min_l3": encoder_values[:, -3:].min(dim=1, keepdim=True).values,
        "l1": encoder_values[:, -1:].mean(dim=1, keepdim=True)
    }

    def compute_metrics(data: torch.Tensor) -> dict:
        """
        Compute iAUC metrics for a given data tensor (predictions or targets) 
        using all baselines.
        
        Args:
            data: Tensor for which to compute iAUC (predictions or targets).                
        Returns:
            A dictionary of metrics with keys like 'iauc_l2' and 'clipped_iauc_l2'.
        """
        metrics = {}
        for key, baseline in baselines.items():
            # Normal iAUC: always allow negative values.
            metrics[f"iauc_{key}"] = _compute_iauc(
                data, baseline, allow_negative=True
            )
            # Calculate the clipped version
            metrics[f"clipped_iauc_{key}"] = _compute_iauc(
                data, baseline, allow_negative=False
            )
        return metrics

    # Compute metrics for predictions.
    pred_metrics = compute_metrics(predictions)    
    gt_metrics = compute_metrics(targets)
    
    return {
        "predictions": pred_metrics,
        "ground_truth": gt_metrics
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
        self.metrics_data = {}  # Store all metrics data
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

    def plot_metric_scatter(self, metric_type: str, metrics_data: dict, metric_correlations: dict, max_points: int = 10000):
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
            metrics_to_plot += ["clipped_iauc_l2", "clipped_iauc_l3", "clipped_iauc_min_l2", "clipped_iauc_min_l3", "clipped_iauc_l1"]
            n_rows, n_cols = 2, 5
            figsize = (25, 10)
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

        if metrics_data is None:
            logger.warning(f"No data available for metric type: {metric_type}")
            return None
        
        metrics_data = metrics_data[metric_type]

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
            if metric_key in metric_correlations:
                corr = metric_correlations[metric_key].item()
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
        self.metrics_data = {
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


        # Combine the metrics into one dictionary with a prefix
        prefix = f"{self.mode}_"
        metric_correlations = {}
        for metric_dict in [iauc_correlations, auc_correlations, raw_cgm_correlations]:
            for k, v in metric_dict.items():
                metric_correlations[f"{prefix}{k}"] = v


        # Create and log scatter plots for each metric type
        # if hasattr(self, 'trainer') and self.trainer is not None:
        for metric_type in ['iauc', 'auc', 'cgm']:
            scatter_fig = self.plot_metric_scatter(metric_type, self.metrics_data, metric_correlations)
            if scatter_fig:
                self.trainer.logger.experiment.log({
                    f"{self.mode}_{metric_type}_scatter": wandb.Image(scatter_fig)
                })
                plt.close(scatter_fig)

        return metric_correlations

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.mode == "val":
            self.trainer = trainer  # Store trainer reference for plotting
            final = self._compute_metrics()
            for key, value in final.items():
                pl_module.log(key, value, prog_bar=True, sync_dist=True)

            # Plot predictions
            self.plot_predictions(trainer, pl_module)

            self.final_metrics = final # Used to reference the metrics post train/test runs
            self.reset_metrics()
            

    def on_test_epoch_end(self, trainer, pl_module):
        if self.mode == "test":
            self.trainer = trainer  # Store trainer reference for plotting
            final = self._compute_metrics()
            for key, value in final.items():
                pl_module.log(key, value, prog_bar=True, sync_dist=True)
                
            # Plot predictions
            self.plot_predictions(trainer, pl_module)
                
            self.final_metrics = final # Used to reference the metrics post train/test runs
            self.reset_metrics()