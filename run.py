import os

# Enable expandable segments for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from loguru import logger
import click
import wandb
import dataclasses

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

from config import Config
from datasets import get_time_series_dataloaders

from model import PPGRTemporalFusionTransformer

from metrics import PPGRMetricsCallback
from loss import build_loss
from utils import create_click_options, setup_experiment_name, initialize_wandb, override_config_from_wandb
from utils import set_random_seeds



def build_callbacks(config: Config) -> (list, PPGRMetricsCallback):
    """Construct and return all Lightning callbacks along with the test metrics callback."""
    early_stop_callback = EarlyStopping(
        monitor=config.early_stop_monitor_metric,
        min_delta=config.early_stop_min_delta,
        patience=config.early_stop_patience,
        verbose=False,
        mode=config.early_stop_monitor_metric_mode
    )
    rich_model_summary = RichModelSummary(max_depth=2)
    rich_progress_bar = RichProgressBar()
    lr_logger = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True,
        log_weight_decay=True
        )
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
    
    if config.profiler:
        profiler = "simple"
    else:
        profiler = None
    
    return pl.Trainer(
        profiler=profiler,
        # precision="bf16",
        max_epochs=config.max_epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks,
        logger=wandb_logger,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.trainer_log_every_n_steps,
    )

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
    set_random_seeds(config.random_seed)

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
        
        # Model hyperparameters
        hidden_size=config.hidden_size,
        lstm_layers=config.lstm_layers,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=config.num_quantiles,
        
        share_single_variable_networks=config.share_single_variable_networks,
        use_transformer_variable_selection_networks=config.use_transformer_variable_selection_networks,
        
        use_transformer_encoder_decoder_layers=config.use_transformer_encoder_decoder_layers,
        transformer_encoder_decoder_num_heads=config.transformer_encoder_decoder_num_heads,
        transformer_encoder_decoder_num_layers=config.transformer_encoder_decoder_num_layers,
        transformer_encoder_decoder_hidden_size=config.transformer_encoder_decoder_hidden_size,
        
        
        enforce_quantile_monotonicity=config.enforce_quantile_monotonicity,
        
        # Loss function
        loss=build_loss(config),        
        
        # Optimizer hyperparameters
        experiment_config=config,

        
        # Logging
        #log_interval=config.trainer_log_interval,
    )
    logger.info(f"Number of parameters in network: {tft_model.size() / 1e3:.1f}k")

    # Build callbacks.
    callbacks, ppgr_metrics_test_callback = build_callbacks(config)

    # Build the trainer.
    trainer = build_trainer(config, callbacks, wandb_logger)

    # Start training.
    trainer.fit(tft_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Run testing.
    trainer.test(tft_model, dataloaders=test_loader, ckpt_path="best")

    # Print final test metrics.
    final_test_metrics = ppgr_metrics_test_callback.final_metrics
    print("Final Test Metrics:")
    for metric_name, metric_value in final_test_metrics.items():
        print(metric_name, metric_value.item())


if __name__ == "__main__":
    main()
