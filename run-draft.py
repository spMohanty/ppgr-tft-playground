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


from config import Config
from datasets import get_cached_time_series_dataloaders
from model import PPGRTemporalFusionTransformer

from metrics import PPGRMetricsCallback

from utils import create_click_options


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
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
    ppgr_metrics_val_callback = PPGRMetricsCallback(mode="val", 
                                                    disable_all_plots=config.disable_all_plots)
    ppgr_metrics_test_callback = PPGRMetricsCallback(mode="test", 
                                                    disable_all_plots=config.disable_all_plots)

    # Initialize callbacks
    callbacks = [   
                    lr_logger, 
                    early_stop_callback, 
                    ppgr_metrics_val_callback, 
                    ppgr_metrics_test_callback
                ]
    
    if not config.disable_checkpoints:
        logger.info(f"checkpoint directory: {config.checkpoint_dir}")
        # Instantiate the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor=config.checkpoint_monitor_metric,
            mode=config.checkpoint_monitor_metric_mode,
            save_top_k=config.checkpoint_top_k,  # Save only the best model.
            dirpath=config.checkpoint_dir,
            filename="{epoch}-{val_loss:.4f}" 
        )
        callbacks.append(checkpoint_callback)
        

    # Build the PyTorch Lightning trainer with all callbacks
    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=config.max_epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks,
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
