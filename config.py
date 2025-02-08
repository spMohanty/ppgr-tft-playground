from dataclasses import dataclass
from rich import print as rprint
from rich.pretty import pprint

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

    # Experiment configuration
    allow_negative_iauc_values: bool = True

    # Data slicing parameters
    max_encoder_length: int = 8 * 4  # encoder window length (e.g., 32)
    max_prediction_length: int = 2 * 4  # prediction horizon (e.g., 8)
    validation_percentage: float = 0.1
    test_percentage: float = 0.1
    dataset_cache_dir: str = "/scratch/mohanty/food/ppgr/datasets-cache"
    no_data_cache: bool = False

    # DataLoader parameters
    batch_size: int = 256 
    num_workers: int = 8

    # Model hyperparameters
    learning_rate: float = 1e-3
    hidden_size: int = 256
    lstm_layers: int = 1
    attention_head_size: int = 4
    dropout: float = 0.15
    hidden_continuous_size: int = 128

    # Trainer parameters
    max_epochs: int = 200  # Early stopping will likely kick in before this.
    gradient_clip_val: float = 0.1
    val_check_interval: float = 0.25
    lr_weight_decay: float = 0.05
    reduce_lr_on_plateau_reduction: float = 10
    reduce_lr_on_plateau_patience: int = 3
    
    loss: str = "QuantileLoss" # "QuantileLoss" or "RMSE"

    # Early stopping parameters
    early_stop_monitor_metric: str = "val_loss"
    early_stop_monitor_metric_mode: str = "min"
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-4

    # Checkpoint parameters
    disable_checkpoints: bool = False
    checkpoint_monitor_metric: str = "val_loss"
    checkpoint_monitor_metric_mode: str = "min"
    checkpoint_dir: str = "/scratch/mohanty/checkpoints/tft-ppgr-2025"
    checkpoint_top_k: int = 5

    # Performance parameters
    disable_all_plots: bool = False


if __name__ == "__main__":
    config = Config()
    rprint("[bold blue]Experiment Configuration:[/bold blue]")
    pprint(config)