from dataclasses import dataclass, fields

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

    # Experiment Configuration
    allow_negative_iauc_values: bool = True
    

    # Data slicing parameters
    max_encoder_length: int = 8 * 4  # encoder window length
    max_prediction_length: int = int(2 * 4)  # prediction horizon (2.5 hours)
    validation_percentage: float = 0.1
    test_percentage: float = 0.1
    dataset_cache_dir: str = "/scratch/mohanty/food/ppgr/datasets-cache"
    no_data_cache: bool = False

    # DataLoader parameters
    batch_size: int = 256
    num_workers: int = 4

    # Model hyperparameters
    learning_rate: float = 1e-2
    hidden_size: int = 256
    attention_head_size: int = 4
    dropout: float = 0.15
    hidden_continuous_size: int = 128

    # Trainer parameters
    max_epochs: int = 200 # early stopping will kick in before this
    gradient_clip_val: float = 0.1

    # Early stopping parameters
    early_stop_monitor_metric: str = "val_loss"
    early_stop_monitor_metric_mode: str = "min"
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-4

    # Logger parameters
    
    # Checkpoint parameters
    disable_checkpoints: bool = False
    checkpoint_monitor_metric: str = "val_loss"
    checkpoint_monitor_metric_mode: str = "min"
    checkpoint_dir: str = "/scratch/mohanty/checkpoints/tft-ppgr-2025"
    checkpoint_top_k: int = 5
    
    # Performance 
    disable_all_plots: bool = False
    