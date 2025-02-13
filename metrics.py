import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from loguru import logger
import wandb

import lightning.pytorch as pl
from torchmetrics.regression import PearsonCorrCoef
from typing import Any, Dict, Tuple

# -----------------------------------------------------------------------------
# Helper Functions for Metrics
# -----------------------------------------------------------------------------
def get_baselines(encoder_values: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute various baseline values from encoder context.
    """
    baselines = {
        "l2": encoder_values[:, -2:].mean(dim=1, keepdim=True),
        "l3": encoder_values[:, -3:].mean(dim=1, keepdim=True),
        "min_l2": encoder_values[:, -2:].min(dim=1, keepdim=True).values,
        "min_l3": encoder_values[:, -3:].min(dim=1, keepdim=True).values,
        "l1": encoder_values[:, -1:].mean(dim=1, keepdim=True)
    }
    return baselines

def _compute_iauc(predicted: torch.Tensor, baseline: torch.Tensor, allow_negative: bool = False) -> torch.Tensor:
    """
    Compute the incremental area under the curve (iAUC) using the trapezoidal rule.
    """
    incremental = predicted - baseline
    if not allow_negative:
        # Use a zeros tensor on the same device as `incremental`
        incremental = torch.maximum(incremental, torch.zeros_like(incremental))
    return torch.trapz(incremental.float(), dim=1)

def compute_iauc_metrics(
    targets: torch.Tensor, 
    predictions: torch.Tensor, 
    encoder_values: torch.Tensor
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute iAUC metrics for both predictions and ground truth.
    """
    baselines = get_baselines(encoder_values)

    def compute_metrics(data: torch.Tensor) -> Dict[str, torch.Tensor]:
        metrics = {}
        for key, baseline in baselines.items():
            metrics[f"iauc_{key}"] = _compute_iauc(data, baseline, allow_negative=True)
            metrics[f"clipped_iauc_{key}"] = _compute_iauc(data, baseline, allow_negative=False)
        return metrics

    return {
        "predictions": compute_metrics(predictions),
        "ground_truth": compute_metrics(targets)
    }

def compute_auc_metrics(
    targets: torch.Tensor, 
    predictions: torch.Tensor, 
    encoder_values: torch.Tensor
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute the AUC metrics for predictions and ground truth. Note that computing iAUC 
    with a baseline of zero is equivalent to computing AUC.
    """
    auc_pred = _compute_iauc(predictions, baseline=0, allow_negative=True)
    auc_ground_truth = _compute_iauc(targets, baseline=0, allow_negative=True)
    
    return {
        "predictions": {"auc": auc_pred},
        "ground_truth": {"auc": auc_ground_truth}
    }

def compute_metric_correlations(metrics: Dict[str, Dict[str, torch.Tensor]], 
                                correlation_calc_fn: Any) -> Dict[str, torch.Tensor]:
    """
    Compute the Pearson correlation between predictions and ground truth for each metric.
    """
    correlations = {}
    for metric_name in metrics["predictions"]:
        pred_vals = metrics["predictions"][metric_name]
        true_vals = metrics["ground_truth"][metric_name]
        correlations[f"{metric_name}_corr"] = correlation_calc_fn(pred_vals, true_vals)
    return correlations

# -----------------------------------------------------------------------------
# PPGR Metrics Callback
# -----------------------------------------------------------------------------
class PPGRMetricsCallback(pl.Callback):
    def __init__(self, mode: str = "val", max_prediction_length: int = 12 * 4, evaluation_horizon_length: int = 2 * 4, 
                 num_plots: int = 6, disable_all_plots: bool = False):
        """
        Custom Lightning callback to compute, log, and plot PPGR metrics.
        
        Args:
            mode (str): Either "val" or "test". Determines which hooks are active.
            num_plots (int): Number of random samples to plot.
            disable_all_plots (bool): If True, disables all plotting.
        """
        super().__init__()
        self.mode = mode
        self.max_prediction_length = max_prediction_length
        self.evaluation_horizon_length = evaluation_horizon_length
        
        self.num_plots = num_plots
        self.disable_all_plots = disable_all_plots
        
        
        
        self.correlation_calc_function = PearsonCorrCoef()
        self.final_metrics = {}
        self.plot_indices = None  # Random indices for plotting.
        self.metrics_data = {}    # Container for metrics data.
        self.reset_metrics()

    def reset_metrics(self):
        self.all_point_predictions = []
        self.all_quantile_predictions = []
        self.all_targets = []
        self.all_encoder_targets = []
        self.raw_batch_data = []
        self.raw_prediction_outputs = []

    def _process_batch(self, pl_module, batch, outputs):
        past_data, _ = batch
        predictions = outputs["prediction"]
        
        if predictions.shape[-1] == 1:
            point_forecast = predictions[:, :, 0]
        else:
            median_quantile_index = predictions.shape[-1] // 2  # Calculate median index dynamically
            point_forecast = predictions[:, :, median_quantile_index]
        self.all_point_predictions.append(point_forecast)
        self.all_quantile_predictions.append(predictions)
        self.all_targets.append(past_data["decoder_target"])
        self.all_encoder_targets.append(past_data["encoder_target"])
        self.raw_batch_data.append(batch)
        self.raw_prediction_outputs.append(outputs)

    def _get_last_batch_output(self, pl_module) -> Tuple[Any, Any]:
        """Retrieve the last batch output based on the current mode."""
        if self.mode == "train":
            return pl_module.training_batch_full_outputs[-1]
        elif self.mode == "val":
            return pl_module.validation_batch_full_outputs[-1]
        elif self.mode == "test":
            return pl_module.test_batch_full_outputs[-1]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.mode == "val":
            _, batch_outputs = self._get_last_batch_output(pl_module)
            self._process_batch(pl_module, batch, batch_outputs)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.mode == "test":
            _, batch_outputs = self._get_last_batch_output(pl_module)
            self._process_batch(pl_module, batch, batch_outputs)
            
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.mode == "train":
            _, batch_outputs = self._get_last_batch_output(pl_module)
            self._process_batch(pl_module, batch, batch_outputs)

    def plot_predictions(self, pl_module, batch_index_to_use: int = 0):
        if self.disable_all_plots:
            return

        past_data, _ = self.raw_batch_data[batch_index_to_use]
        prediction_outputs = self.raw_prediction_outputs[batch_index_to_use]
        if self.plot_indices is None:
            batch_size = past_data["encoder_target"].shape[0]
            self.plot_indices = random.sample(range(batch_size), min(self.num_plots, batch_size))
            logger.info(f"Selected indices for plotting: {self.plot_indices}")

        for idx in self.plot_indices:
            fig = pl_module.plot_prediction(past_data, prediction_outputs, idx)
            self.trainer.logger.experiment.log({f"predictions_sample_{idx}": wandb.Image(fig)})

    def plot_metric_scatter(self, metric_type: str, metrics_data: dict, metric_correlations: dict, max_points: int = 10000, logger_prefix: str = ""):
        """
        Create scatter plots comparing predicted vs. actual values for a given metric type.
        """
        import seaborn as sns  # Imported here to avoid a global dependency if plotting is disabled.

        if self.disable_all_plots:
            return None

        if metric_type == 'iauc':
            metrics_to_plot = [
                "iauc_l2", "iauc_l3", "iauc_min_l2", "iauc_min_l3", "iauc_l1",
                "clipped_iauc_l2", "clipped_iauc_l3", "clipped_iauc_min_l2", "clipped_iauc_min_l3", "clipped_iauc_l1"
            ]
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

        metrics_subset = metrics_data.get(metric_type, {})
        for idx, metric in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break

            pred = metrics_subset["predictions"][metric].detach().cpu().to(torch.float32).numpy()
            true = metrics_subset["ground_truth"][metric].detach().cpu().to(torch.float32).numpy()

            # Random subsample if too many points
            if len(pred) > max_points:
                indices = np.random.choice(len(pred), max_points, replace=False)
                pred = pred[indices]
                true = true[indices]

            sns.scatterplot(x=true, y=pred, s=0.75, alpha=0.5, ax=axes[idx])
            axes[idx].set_xlabel(f'True {title_prefix}')
            axes[idx].set_ylabel(f'Predicted {title_prefix}')
            axes[idx].set_title(f'{metric} Scatter Plot\n(n={len(pred):,} points)')

            metric_key = f"{logger_prefix}_{metric}_corr"
            if metric_key in metric_correlations:
                corr = metric_correlations[metric_key].item()
                axes[idx].text(0.05, 0.95, f'r = {corr:.3f}',
                               transform=axes[idx].transAxes,
                               bbox=dict(facecolor='white', alpha=0.8))

        # Remove any extra axes
        for ax in axes[len(metrics_to_plot):]:
            fig.delaxes(ax)

        plt.tight_layout()
        plt.close(fig)
        return fig
    
    def compute_monotonicity_violation_metrics(self, outputs: torch.Tensor):
        """
        Computes three monotonicity metrics for quantile outputs.
        
        Args:
            outputs (torch.Tensor): Tensor of shape [batch, forecast_horizon, num_quantiles].
                                    The quantiles are assumed to be ordered along the last dimension.
                                    
        Returns:
            dict: A dictionary containing the following metrics:
                - "mean_violation" (torch.Tensor): The mean monotonicity violation averaged over batch and forecast horizon.
                - "violation_rate" (torch.Tensor): The fraction of quantile pairs (per forecast point) that violate monotonicity.
                - "max_violation" (torch.Tensor): The maximum violation observed across the batch and forecast horizon.
        """
        # Compute differences between adjacent quantiles.
        # For each forecast point, we expect: q1 <= q2 <= ... <= q_num_quantiles.
        # diff > 0 indicates a violation (i.e., q_j > q_{j+1}).
        diff = outputs[..., :-1] - outputs[..., 1:]  # shape: [batch, horizon, num_quantiles - 1]

        # Only positive differences are violations.
        violation = torch.relu(diff)  # same shape as diff

        # Mean violation: sum the violation values for each forecast point and average over all points.
        # This gives an idea of the average "amount" by which quantile order is violated.
        mean_violation = violation.sum(dim=-1).mean()  # scalar

        # Violation rate: count how many adjacent pairs are violating the order and average.
        # This gives a fraction between 0 and 1.
        total_pairs = outputs.shape[-1] - 1
        violation_rate = ((diff > 0).float().sum(dim=-1) / total_pairs).mean()  # scalar

        # Maximum violation: the largest single violation value observed.
        max_violation = violation.max()  # scalar

        return {
            "mean_violation": mean_violation,
            "violation_rate": violation_rate,
            "max_violation": max_violation
        }
    

    def _compute_metrics(self, pl_module: Any = None, evaluation_horizon_length: int = 2 * 4, plot: bool = True, log: bool = True) -> Dict[str, torch.Tensor]:
        # Use a consistent predict for all logs and plots
        logger_prefix = f"{self.mode}_eh{evaluation_horizon_length}"

        # Gather all predictions, targets, and encoder values needed to compute metrics
        all_preds = torch.cat(self.all_point_predictions, dim=0)
        all_quantile_preds = torch.cat(self.all_quantile_predictions, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)
        all_encoder_targets = torch.cat(self.all_encoder_targets, dim=0)
        
        # If evaluation horizon length is not the same as the max prediction length, we need to truncate the predictions and targets
        if evaluation_horizon_length != self.max_prediction_length:
            all_preds = all_preds[:, :evaluation_horizon_length]
            all_quantile_preds = all_quantile_preds[:, :evaluation_horizon_length, :]
            all_targets = all_targets[:, :evaluation_horizon_length]
            all_encoder_targets = all_encoder_targets[:, :evaluation_horizon_length]
        
        iauc_metrics = compute_iauc_metrics(all_targets, all_preds, all_encoder_targets)
        auc_metrics = compute_auc_metrics(all_targets, all_preds, all_encoder_targets)
        raw_cgm_metrics = {
            "predictions": {"raw_cgm": all_preds.flatten()},
            "ground_truth": {"raw_cgm": all_targets.flatten()}
        }
        
        
        monotonicity_violation_metrics = self.compute_monotonicity_violation_metrics(all_quantile_preds)
        metric_monotonicity_violations = { f"{logger_prefix}_q-monotonicity_violation_{key}": value
            for key, value in monotonicity_violation_metrics.items()
        }
        
        # Save metrics for plotting
        all_metrics_data = {
            'iauc': iauc_metrics,
            'auc': auc_metrics,
            'cgm': raw_cgm_metrics,
            'q-monotonicity_violation': metric_monotonicity_violations
        }

        # Make sure the correlation function is on the same device as the metrics
        self.correlation_calc_function = self.correlation_calc_function.to(all_targets.device)
        
        # Calculate the correlations
        iauc_corr = compute_metric_correlations(iauc_metrics, self.correlation_calc_function)
        auc_corr = compute_metric_correlations(auc_metrics, self.correlation_calc_function)
        cgm_corr = compute_metric_correlations(raw_cgm_metrics, self.correlation_calc_function)


        metric_correlations = {
            f"{logger_prefix}_{key}": value
            for corr_dict in [iauc_corr, auc_corr, cgm_corr]
            for key, value in corr_dict.items()
        }
        all_metrics_data.update(metric_correlations)

        if plot:
            # Create and log scatter plots for each metric type
            for metric_type in ['iauc', 'auc', 'cgm']:
                scatter_fig = self.plot_metric_scatter(metric_type, all_metrics_data, metric_correlations, logger_prefix=logger_prefix)
                if scatter_fig:
                    self.trainer.logger.experiment.log({
                        f"{logger_prefix}_{metric_type}_scatter": wandb.Image(scatter_fig)
                    })
                    plt.close(scatter_fig)
                    
            # Plot predictions
            self.plot_predictions(pl_module)
            
        if log:            
            # Log all correlation metrics
            for key, value in metric_correlations.items():
                pl_module.log(key, value, prog_bar=True, sync_dist=True)
                
            # log all monotonicity violation metrics
            for key, value in metric_monotonicity_violations.items():
                pl_module.log(key, value, prog_bar=True, sync_dist=True)
            

            
        return metric_correlations

    def _handle_epoch_end(self, trainer, pl_module, plot: bool = True):
        """
        Common logic to run at the end of an epoch (either validation or test).
        """
        self.trainer = trainer
        self.final_metrics = self._compute_metrics(pl_module, evaluation_horizon_length=self.max_prediction_length, plot=plot)
        self.final_metrics_at_evaluation_horizon = None
        
        if self.evaluation_horizon_length != self.max_prediction_length:
            # Call the compute metrics again with the expected evaluation horizon length
            self.final_metrics_at_evaluation_horizon = self._compute_metrics(pl_module, evaluation_horizon_length=self.evaluation_horizon_length, plot=plot)
        
        self.reset_metrics()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.mode == "val":
            self._handle_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.mode == "test":
            self._handle_epoch_end(trainer, pl_module)
            
    def on_train_epoch_end(self, trainer, pl_module):
        if self.mode == "train":
            # disable plotting for training for performance reasons
            # todo: make this configurable - could be helpful for debugging 
            self._handle_epoch_end(trainer, pl_module, plot=False)

if __name__ == "__main__":
    # Define dummy classes to mimic Trainer and Module behavior.
    class DummyExperiment:
        def log(self, log_dict):
            print("Experiment log:", log_dict)

    class DummyLogger:
        def __init__(self):
            self.experiment = DummyExperiment()

    class DummyTrainer:
        def __init__(self):
            self.logger = DummyLogger()

    class DummyModule:
        def __init__(self):
            # These lists mimic the full outputs stored during training/validation.
            self.validation_batch_full_outputs = []
            self.test_batch_full_outputs = []

        def log(self, key, value, prog_bar, sync_dist):
            print(f"Logging {key}: {value} (prog_bar={prog_bar}, sync_dist={sync_dist})")

        def plot_prediction(self, past_data, prediction_outputs, idx):
            # Create a dummy plot for the given index.
            fig, ax = plt.subplots()
            encoder_values = past_data["encoder_target"][idx].detach().cpu().to(torch.float32).numpy()
            decoder_target = past_data["decoder_target"][idx].detach().cpu().to(torch.float32).numpy()
            ax.plot(encoder_values, label="encoder_target")
            ax.plot(decoder_target, label="decoder_target")
            ax.legend()
            ax.set_title(f"Sample {idx} Prediction Plot")
            return fig

    # Create synthetic data for a single batch.
    batch_size = 8
    encoder_length = 10
    prediction_length = 5
    num_quantiles = 7  # Ensures index 3 (the median) exists

    # Create dummy past_data with encoder and decoder targets.
    past_data = {
        "encoder_target": torch.randn(batch_size, encoder_length),
        "decoder_target": torch.randn(batch_size, prediction_length)
    }
    future_data = {}  # Dummy future data (not used in metrics)
    batch = (past_data, future_data)

    # Create dummy outputs with predictions of shape (batch_size, prediction_length, num_quantiles).
    predictions = torch.randn(batch_size, prediction_length, num_quantiles)
    outputs = {"prediction": predictions}

    # Instantiate the dummy module and simulate storing a batch output.
    dummy_module = DummyModule()
    dummy_module.validation_batch_full_outputs.append((None, outputs))

    # Instantiate the dummy trainer.
    trainer = DummyTrainer()

    # Instantiate the callback (set num_plots to 3 for brevity).
    callback = PPGRMetricsCallback(mode="val", num_plots=3, disable_all_plots=False)

    # Simulate processing a validation batch.
    callback.on_validation_batch_end(trainer, dummy_module, outputs, batch, batch_idx=0)

    # Simulate end-of-epoch processing (which computes and logs metrics and plots).
    callback.on_validation_epoch_end(trainer, dummy_module)
