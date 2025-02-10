#!/usr/bin/env python
"""
Utility module for experiment setup including:
    - Automatic Click options creation from dataclass configurations.
    - Random seed setting for reproducibility.
    - Symmetric quantile calculation.
    - Experiment naming.
    - Initialization and configuration override for Weights & Biases.
"""

import os
import random
from typing import Any, Callable, List, Type

import click
import numpy as np
import torch
import wandb
from dataclasses import MISSING, fields
from loguru import logger
from lightning.pytorch.loggers import WandbLogger

from config import Config

import torch.nn.functional as F

def create_click_options(config_class: Type[Any]) -> Callable:
    """
    Create a decorator that adds Click options based on the fields
    of a dataclass configuration.

    Args:
        config_class: A dataclass type containing configuration parameters.

    Returns:
        A decorator that can be applied to a Click command function.
    """
    # Mapping for converting dataclass types to click types
    type_mapping = {str: str, int: int, float: float, bool: bool}

    def decorator(f: Callable) -> Callable:
        # Iterate in reverse so that decorators are applied in the correct order.
        for field in reversed(list(fields(config_class))):
            # Determine the Click parameter type.
            field_type = type_mapping.get(field.type, str)

            if field.default is MISSING:
                option_kwargs = {"required": True, "help": "Required parameter"}
            else:
                option_kwargs = {
                    "default": field.default,
                    "show_default": True,
                    "help": f"Default: {field.default}",
                }

            if field_type == bool:
                # Use dual flag syntax: "--field/--no-field" so the default from the config is honored.
                option_declaration = f"--{field.name.replace('_', '-')}/--no-{field.name.replace('_', '-')}"
                option_names = [option_declaration]
                
                # For special cases, like "debug_mode", add an additional alias if needed.
                if field.name == "debug_mode":
                    option_names.append("--debug/--no-debug")
                
                # Pass the actual default from the config to click (will be True/False)
                f = click.option(*option_names, default=field.default, show_default=True,
                                 help=f"Default: {field.default}")(f)
            else:
                # Create both hyphenated and underscored option names.
                hyphen_name = f"--{field.name.replace('_', '-')}"
                underscore_name = f"--{field.name}"
                option_names = [hyphen_name, underscore_name]
                f = click.option(*option_names, type=field_type, **option_kwargs)(f)
        return f

    return decorator

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: The seed value to use (default is 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seeds set to {seed}")


def calculate_symmetric_quantiles(num_quantiles: int) -> List[float]:
    """
    Calculate symmetric quantile levels given the desired total number of quantiles.

    - For num_quantiles == 1, returns [0.5].
    - For an odd number (>1), includes the median (0.5) with the lower half 
      evenly spaced between 0 and 0.5 and the upper half mirrored accordingly.
    - For an even number, returns symmetric pairs around 0.5.

    Args:
        num_quantiles: The total number of quantiles (must be >= 1).

    Returns:
        A sorted list of quantile values between 0 and 1.
    
    Raises:
        ValueError: If num_quantiles is less than 1.
    """
    if num_quantiles < 1:
        raise ValueError("num_quantiles must be at least 1")
    if num_quantiles == 1:
        return [0.5]

    half_count = num_quantiles // 2

    # Generate lower quantiles in the open interval (0, 0.5)
    lower_quantiles = np.linspace(0, 0.5, num=half_count + 1, endpoint=False)[1:]

    if num_quantiles % 2 == 1:
        # Odd: include the median (0.5) and mirror lower quantiles.
        quantiles = np.concatenate((lower_quantiles, [0.5], 1.0 - lower_quantiles[::-1]))
    else:
        # Even: mirror the lower quantiles.
        quantiles = np.concatenate((lower_quantiles, 1.0 - lower_quantiles[::-1]))

    return quantiles.tolist()


def setup_experiment_name(config: Config) -> str:
    """
    Construct an experiment name based on configuration hyperparameters.

    Args:
        config: The configuration object with hyperparameters.

    Returns:
        A string representing the experiment name.
    """
    base_name = config.experiment_name
    experiment_name_parts = [
        base_name,
        f"hs{config.hidden_size}",
        f"ahs{config.attention_head_size}",
        f"hcs{config.hidden_continuous_size}",
        f"d{int(config.dropout * 100)}",
        f"lr{config.learning_rate:.0e}",
        f"bs{config.batch_size}",
    ]
    experiment_name = "-".join(experiment_name_parts)
    logger.debug(f"Experiment name constructed: {experiment_name}")
    return experiment_name


def initialize_wandb(config: Config, experiment_name: str) -> WandbLogger:
    """
    Initialize Weights & Biases (wandb) and return a Lightning WandbLogger.

    This function also sets the WANDB_DIR environment variable and logs
    the wandb run directory.

    Args:
        config: The configuration object.
        experiment_name: The name to assign to the experiment/run.

    Returns:
        An instance of WandbLogger.
    """
    os.makedirs(config.wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = config.wandb_dir
    logger.info(f"WANDB_DIR set to: {os.environ['WANDB_DIR']}")

    # Initialize the wandb run.
    wandb.init(
        project=config.wandb_project,
        name=experiment_name,
        dir=config.wandb_dir,
        settings=wandb.Settings(start_method="thread")
    )
    logger.info(f"Initialized wandb run with directory: {wandb.run.dir}")

    return WandbLogger(project=config.wandb_project, name=experiment_name)


def override_config_from_wandb(config: Config) -> None:
    """
    Override configuration values with wandb configuration parameters during sweeps.

    If wandb.run exists, this function updates the configuration parameters from
    wandb.config and renames the wandb run accordingly.

    Args:
        config: The configuration object to update.
    """
    if hasattr(wandb, 'run') and wandb.run is not None:
        # Update configuration fields if present in wandb.config.
        config.learning_rate = wandb.config.get("learning_rate", config.learning_rate)
        config.hidden_size = wandb.config.get("hidden_size", config.hidden_size)
        config.attention_head_size = wandb.config.get("attention_head_size", config.attention_head_size)
        config.dropout = wandb.config.get("dropout", config.dropout)
        config.hidden_continuous_size = wandb.config.get("hidden_continuous_size", config.hidden_continuous_size)
        config.max_epochs = wandb.config.get("max_epochs", config.max_epochs)
        config.batch_size = wandb.config.get("batch_size", config.batch_size)

        # Update the experiment name based on the new configuration.
        new_experiment_name = setup_experiment_name(config)
        wandb.run.name = new_experiment_name
        wandb.run.save()
        logger.info(f"Configuration overridden from wandb. New experiment name: {new_experiment_name}")
    else:
        logger.warning("wandb.run is not available. Config not overridden.")

def conditional_enforce_quantile_monotonicity(tensor_input: torch.Tensor,
                                                enforce_quantile_monotonicity: bool = True) -> torch.Tensor:
    """
    Enforces quantile monotonicity on the output tensor if requested.

    In some quantile regression setups, the model is designed to predict the _increments_
    between quantiles rather than the absolute quantile values. To obtain the actual quantile
    values, we need to perform the following two steps:
    
      1. Apply a non-negative activation (e.g., softplus) to the raw outputs so that all increments are ≥ 0.
      2. Compute the cumulative sum along the quantile dimension to convert increments into absolute quantile values.
    
    This ensures that the quantile values are monotonically increasing (i.e., no quantile crossing).

    Args:
        tensor_input (torch.Tensor): The raw tensor representing quantile increments.
            Expected shape: [..., num_quantiles]. For example, with an output shape of [70, 8, 7],
            the last dimension corresponds to the 7 quantile increments. 
            This is the output of the previous later, and the output of this should be the final layer.
        enforce_quantile_monotonicity (bool): Flag to enable/disable the monotonicity enforcement.
            If True, the cumulative sum is applied; if False, the raw output is returned as is.

    Returns:
        torch.Tensor: The processed tensor containing monotonic quantile values (if enforcement is enabled)
        or the raw output otherwise.
    """
    if enforce_quantile_monotonicity:
        # Step 1: Ensure the increments are non-negative by applying softplus.
        # This is important because negative increments could lead to non-monotonic quantile values.
        non_negative_increments = F.relu(tensor_input)
        
        # Step 2: Convert increments into absolute quantile values using a cumulative sum.
        # The cumulative sum is computed along the last dimension (assumed to be the quantile dimension).
        output = torch.cumsum(non_negative_increments, dim=-1)
    else:
        output = tensor_input
    
    return output

# Optional: An example entry point for testing these utilities.
if __name__ == "__main__":
    # Example usage (requires a proper Config definition in config.py)
    try:
        # Create a dummy config if needed for testing.
        config = Config()  # Assumes that Config is a dataclass with defaults.
        set_random_seeds(config.seed if hasattr(config, "seed") else 42)
        experiment_name = setup_experiment_name(config)
        logger.info(f"Experiment Name: {experiment_name}")

        # Example: Calculate quantiles
        quantiles = calculate_symmetric_quantiles(5)
        logger.info(f"Calculated quantiles: {quantiles}")

        # Initialize wandb if configured.
        wb_logger = initialize_wandb(config, experiment_name)
        logger.info(f"Initialized WandbLogger: {wb_logger}")

        # Optionally override configuration from wandb.
        override_config_from_wandb(config)

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
