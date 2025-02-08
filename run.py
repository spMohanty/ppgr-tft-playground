import os
import numpy as np
import torch
from loguru import logger
import click
import wandb
import dataclasses

from pytorch_forecasting.metrics import QuantileLoss, RMSE, MultiLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

from config import Config
from datasets import get_time_series_dataloaders
from model import PPGRTemporalFusionTransformer
from metrics import PPGRMetricsCallback
from utils import create_click_options


def setup_experiment_name(config: Config) -> str:
    """Construct an experiment name based on configuration hyperparameters."""
    base_name = config.experiment_name
    experiment_name_parts = [
        base_name,
        f"hs{config.hidden_size}",
        f"ahs{config.attention_head_size}",
        f"hcs{config.hidden_continuous_size}",
        f"d{int(config.dropout * 100)}",
        f"lr{config.learning_rate:.0e}",
        f"bs{config.batch_size}"
    ]
    return "-".join(experiment_name_parts)


def initialize_wandb(config: Config, experiment_name: str) -> WandbLogger:
    """Initialize wandb and return a WandbLogger for Lightning."""
    os.makedirs(config.wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = config.wandb_dir
    logger.info(f"WANDB_DIR: {os.environ['WANDB_DIR']}")

    wandb.init(
        project=config.wandb_project,
        name=experiment_name,
        dir=config.wandb_dir,
        settings=wandb.Settings(start_method="thread")
    )
    logger.info(f"Wandb directory: {wandb.run.dir}")

    return WandbLogger(
        project=config.wandb_project,
        name=experiment_name
    )


def override_config_from_wandb(config: Config) -> None:
    """
    When running a sweep, wandb.config will contain the current run's hyperparameters.
    Override default config values if they exist.
    """
    if hasattr(wandb, 'config') and wandb.run is not None:
        config.learning_rate = wandb.config.get("learning_rate", config.learning_rate)
        config.hidden_size = wandb.config.get("hidden_size", config.hidden_size)
        config.attention_head_size = wandb.config.get("attention_head_size", config.attention_head_size)
        config.dropout = wandb.config.get("dropout", config.dropout)
        config.hidden_continuous_size = wandb.config.get("hidden_continuous_size", config.hidden_continuous_size)
        config.max_epochs = wandb.config.get("max_epochs", config.max_epochs)
        config.batch_size = wandb.config.get("batch_size", config.batch_size)

        # Update experiment name based on sweep parameters.
        new_experiment_name = setup_experiment_name(config)
        wandb.run.name = new_experiment_name
        wandb.run.save()


def build_callbacks(config: Config) -> (list, PPGRMetricsCallback):
    """Construct and return all Lightning callbacks along with the test metrics callback."""
    early_stop_callback = EarlyStopping(
        monitor=config.early_stop_monitor_metric,
        min_delta=config.early_stop_min_delta,
        patience=config.early_stop_patience,
        verbose=False,
        mode=config.early_stop_monitor_metric_mode
    )
    rich_model_summary = RichModelSummary()
    rich_progress_bar = RichProgressBar()
    lr_logger = LearningRateMonitor(logging_interval="step")
    ppgr_metrics_val_callback = PPGRMetricsCallback(mode="val", disable_all_plots=config.disable_all_plots)
    ppgr_metrics_test_callback = PPGRMetricsCallback(mode="test", disable_all_plots=config.disable_all_plots)

    callbacks = [
        rich_model_summary,
        rich_progress_bar,
        lr_logger,
        ppgr_metrics_val_callback,
        ppgr_metrics_test_callback,
        early_stop_callback
    ]

    if not config.disable_checkpoints:
        logger.info(f"Using checkpoint directory: {config.checkpoint_dir}")
        checkpoint_callback = ModelCheckpoint(
            monitor=config.checkpoint_monitor_metric,
            mode=config.checkpoint_monitor_metric_mode,
            save_top_k=config.checkpoint_top_k,
            dirpath=config.checkpoint_dir,
            filename="{epoch}-{val_loss:.4f}"
        )
        callbacks.append(checkpoint_callback)

    return callbacks, ppgr_metrics_test_callback


def build_trainer(config: Config, callbacks: list, wandb_logger: WandbLogger) -> pl.Trainer:
    """Instantiate and return the Lightning Trainer."""
    return pl.Trainer(
        profiler="simple",
        max_epochs=config.max_epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks,
        logger=wandb_logger,
        val_check_interval=config.val_check_interval,
    )

def build_loss(config: Config) -> torch.nn.Module:
    """Build and return the loss function based on the configuration."""
    if config.loss == "QuantileLoss":
        return QuantileLoss()
    elif config.loss == "RMSE":
        return RMSE()
    elif config.loss == "Quantile+RMSE":
        return MultiLoss(metrics = [QuantileLoss(), RMSE()])
    else:
        raise ValueError(f"Invalid loss function: {config.loss}")

@click.command()
@create_click_options(Config)
def main(**kwargs):
    """
    Main training function.
    All CLI arguments are collected and passed to the Config constructor.
    """
    # Initialize configuration from CLI parameters.
    config = Config(**kwargs)

    # Construct a meaningful experiment name.
    experiment_name = setup_experiment_name(config)
    config.experiment_name = experiment_name

    # Initialize wandb and the corresponding Lightning logger.
    wandb_logger = initialize_wandb(config, experiment_name)

    # In debug mode, run only one epoch.
    if config.debug_mode:
        config.max_epochs = 1

    # Override config parameters if running a sweep.
    override_config_from_wandb(config)

    # Log hyperparameters.
    hyperparameters = dataclasses.asdict(config)
    wandb_logger.log_hyperparams(hyperparameters)

    # Set random seeds.
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # Retrieve cached datasets and create dataloaders.
    train_loader, val_loader, test_loader = get_time_series_dataloaders(config, 
                                                                        shuffle_train=True,
                                                                        shuffle_val=False, 
                                                                        shuffle_test=False)

    # Log device info.
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        logger.info("CUDA is not available. Using CPU.")

    # Build the Temporal Fusion Transformer model using the training dataset.
    tft_model = PPGRTemporalFusionTransformer.from_dataset(
        train_loader.dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        loss=build_loss(config),
        log_interval=5,  # Adjust logging frequency as needed.
        optimizer="ranger",
        reduce_on_plateau_patience=4,
    )
    logger.info(f"Number of parameters in network: {tft_model.size() / 1e3:.1f}k")

    # Build callbacks.
    callbacks, ppgr_metrics_test_callback = build_callbacks(config)

    # Build the trainer.
    trainer = build_trainer(config, callbacks, wandb_logger)

    # Start training.
    trainer.fit(tft_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Run testing.
    trainer.test(tft_model, dataloaders=test_loader)

    # Print final test metrics.
    final_test_metrics = ppgr_metrics_test_callback.final_metrics
    print("Final Test Metrics:")
    for metric_name, metric_value in final_test_metrics.items():
        print(metric_name, metric_value.item())


if __name__ == "__main__":
    main()
