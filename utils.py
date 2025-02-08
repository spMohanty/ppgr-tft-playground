from dataclasses import dataclass, fields

import click
import random
import numpy as np
import torch

import os
from loguru import logger

import wandb

from config import Config
from lightning.pytorch.loggers import WandbLogger

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


def set_random_seeds(seed: int = 42):
    """Helper function to set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_symmetric_quantiles(num_quantiles: int) -> list:
    """
    Calculate symmetric quantile levels given the desired total number of quantiles.

    - For num_quantiles == 1, returns [0.5].
    - For an odd number (>1), includes the median (0.5) with the lower half 
      evenly spaced between 0 and 0.5 and the upper half mirrored accordingly.
    - For an even number, returns symmetric pairs around 0.5.

    Parameters:
        num_quantiles (int): The total number of quantiles (must be >= 1).

    Returns:
        list of float: Sorted quantile values between 0 and 1.
    """
    if num_quantiles < 1:
        raise ValueError("num_quantiles must be at least 1")
    if num_quantiles == 1:
        return [0.5]
    
    # Calculate the number of quantiles in the lower half.
    half = num_quantiles // 2
    
    # Generate lower quantiles in the open interval (0, 0.5).
    # We use endpoint=False to avoid including zero, and then slice off the first element.
    lower_quantiles = np.linspace(0, 0.5, num=half + 1, endpoint=False)[1:]
    
    # For an odd number of quantiles, include the median (0.5).
    if num_quantiles % 2 == 1:
        quantiles = np.concatenate((lower_quantiles, [0.5], 1.0 - lower_quantiles[::-1]))
    else:
        quantiles = np.concatenate((lower_quantiles, 1.0 - lower_quantiles[::-1]))
    
    return quantiles.tolist()

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