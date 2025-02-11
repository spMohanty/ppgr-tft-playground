from dataclasses import dataclass, asdict
from rich import print as rprint
from rich.pretty import pprint

from typing import Tuple

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
    batch_size: int = 1024 * 3
    num_workers: int = 16

    # Model hyperparameters
    
    hidden_size: int = 256
    dropout: float = 0.15
    num_quantiles: int = 7 # 7 or 13 or any odd number > 3
    attention_head_size: int = 4
    hidden_continuous_size: int = 32
    
    share_single_variable_networks: bool = True
    
    use_lstm_encoder_decoder_layers: bool = False # Disabling LSTM based encoder-decoder layers by default
    lstm_layers: int = 1
    
    use_transformer_encoder_decoder_layers: bool = True
    transformer_encoder_decoder_num_heads: int = 4
    transformer_encoder_decoder_num_layers: int = 4
    transformer_encoder_decoder_hidden_size: int = 32
    
    enforce_quantile_monotonicity: bool = False

    # Trainer parameters
    max_epochs: int = 30  # Early stopping will likely kick in before this.
    gradient_clip_val: float = 0.1
    
    optimizer: str = "adamw" # cannot change this atm 
    optimizer_weight_decay: float = 0.05

    lr_scheduler: str = "onecycle" # cannot change this atm 
    
    learning_rate: float = 1e-4
    lr_scheduler_max_lr_multiplier: float = 1.5 # make the lr 1.5 (multiplier) times in the first 5% of the steps, then slowly decrease until the end of all the epochs
    lr_scheduler_pct_start: float = 0.1 # should keep it around upto 2.5 epochs for max 30 epochs - kind of a warm up
    lr_scheduler_anneal_strategy: str = "cos"
    lr_scheduler_cycle_momentum: bool = False
    
    # Precision parameters
    training_precision: str = "bf16" # "bf16" or "fp32" (output_layer of the model still runs in fp32)
    
    
    val_check_interval: float = 0.5
    trainer_log_every_n_steps: int = 1
    
    loss: str = "QuantileLoss" # "QuantileLoss" or "ApproximateCRPS" or "RMSE"

    # Early stopping parameters
    early_stop_enabled: bool = False
    early_stop_monitor_metric: str = "val_loss"
    early_stop_monitor_metric_mode: str = "min"
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-6

    # Checkpoint parameters
    disable_checkpoints: bool = True
    checkpoint_monitor_metric: str = "val_loss"
    checkpoint_monitor_metric_mode: str = "min"
    checkpoint_dir: str = "/scratch/mohanty/checkpoints/tft-ppgr-2025"
    checkpoint_top_k: int = 5

    # Performance parameters
    disable_all_plots: bool = True
    profiler:bool = False


if __name__ == "__main__":
    config = Config()
    rprint("[bold blue]Experiment Configuration:[/bold blue]")
    pprint(config)
    config.quantiles = "asomething"
    pprint(config)
    
    print(asdict(config))