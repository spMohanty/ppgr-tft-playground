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
    ppgr_metrics_val_callback = PPGRMetricsCallback(mode="val", 
                                                    max_prediction_length=config.max_prediction_length,
                                                    evaluation_horizon_length=config.evaluation_horizon_length,
                                                    num_plots=config.number_of_plots_in_metrics_callback,
                                                    disable_all_plots=config.disable_all_plots)
    ppgr_metrics_test_callback = PPGRMetricsCallback(mode="test", 
                                                     max_prediction_length=config.max_prediction_length,
                                                     evaluation_horizon_length=config.evaluation_horizon_length,
                                                     num_plots=config.number_of_plots_in_metrics_callback,
                                                     disable_all_plots=config.disable_all_plots)

    callbacks = [
        rich_model_summary,
        rich_progress_bar,
        lr_logger,
        ppgr_metrics_val_callback,
        ppgr_metrics_test_callback,
    ]
    
    if config.early_stop_enabled:
        callbacks.append(early_stop_callback)

    if not config.disable_checkpoints:
        logger.info(f"Disable checkpoints is ??? {config.disable_checkpoints}")
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
        precision=config.training_precision,
        max_epochs=config.max_epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks,
        logger=wandb_logger,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.trainer_log_every_n_steps,
        enable_checkpointing=not config.disable_checkpoints,
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
    if config.debug:
        config.max_epochs = 1
        # config.include_food_covariates = True
        # config.include_food_covariates_from_horizon = False

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
        
    # Check if Flash Attention is available.
    if torch.cuda.is_available():
        if torch.backends.cuda.flash_sdp_enabled():
            logger.success("Flash Attention available.")
        else:
            logger.error("Flash Attention not available.")
            
        if torch.backends.cuda.mem_efficient_sdp_enabled():
            logger.success("Mem Efficient Attention available.")
        else:
            logger.error("Mem Efficient Attention not available.")
                
    # Build the Temporal Fusion Transformer model using the training dataset.
    tft_model = PPGRTemporalFusionTransformer.from_dataset(
        train_loader.dataset,
        
        # Model hyperparameters
        hidden_size=config.hidden_size,
        hidden_continuous_size=config.hidden_continuous_size,
        attention_head_size=config.attention_head_size,
        
        dropout=config.dropout,
        output_size=config.num_quantiles,
        
        # Optimizer and learning rate scheduler settings
        optimizer = config.optimizer,
        lr_scheduler = config.lr_scheduler,
        lr_scheduler_max_lr_multiplier = config.lr_scheduler_max_lr_multiplier,
        lr_scheduler_pct_start = config.lr_scheduler_pct_start,
        lr_scheduler_anneal_strategy = config.lr_scheduler_anneal_strategy,
        lr_scheduler_cycle_momentum = config.lr_scheduler_cycle_momentum,
        learning_rate = config.learning_rate,
        optimizer_weight_decay = config.optimizer_weight_decay,

        # Variable Selection Networks
        variable_selection_network_n_heads = config.variable_selection_network_n_heads,
        share_single_variable_networks = config.share_single_variable_networks,
                
        # Transformer encoder/decoder configuration
        transformer_encoder_decoder_num_heads = config.transformer_encoder_decoder_num_heads,
        transformer_encoder_decoder_hidden_size = config.transformer_encoder_decoder_hidden_size,
        transformer_encoder_decoder_num_layers = config.transformer_encoder_decoder_num_layers,        
        
        # Additional inputs and settings
        max_encoder_length = config.max_encoder_length,
        enforce_quantile_monotonicity = config.enforce_quantile_monotonicity,                
        
        # Loss function
        loss=build_loss(config),        
                
        # Logging
        #log_interval=config.trainer_log_interval,
    )
    logger.info(f"Number of parameters in network: {tft_model.size() / 1e3:.1f}k")

    # Build callbacks.
    callbacks, ppgr_metrics_test_callback = build_callbacks(config)
    
    logger.info(f"Callbacks: {callbacks}")

    # Build the trainer.
    trainer = build_trainer(config, callbacks, wandb_logger)
    
    # Start training.
    trainer.fit(tft_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Run testing.
    ckpt_path = "best" if not config.disable_checkpoints else None
    trainer.test(tft_model, dataloaders=test_loader, ckpt_path=ckpt_path)

    # Print final test metrics.
    final_test_metrics = ppgr_metrics_test_callback.final_metrics
    print("Final Test Metrics:")
    for metric_name, metric_value in final_test_metrics.items():
        print(metric_name, metric_value.item())


if __name__ == "__main__":
    main()
