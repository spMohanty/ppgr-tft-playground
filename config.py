from dataclasses import dataclass, asdict
from rich import print as rprint
from rich.pretty import pprint

from typing import Tuple, Any

@dataclass
class Config:
    # General configuration
    experiment_name: str = "tft-ppgr-2025-fo"
    random_seed: int = 42
    dataset_version: str = "v0.4"
    debug_mode: bool = False

    # WandB configuration
    wandb_project: str = "tft-ppgr-2025-debug"
    wandb_dir: str = "/scratch/mohanty/wandb"

    # Experiment configuration
    allow_negative_iauc_values: bool = True

    # Data slicing parameters
    max_encoder_length: int = 8 * 4  # encoder window length (e.g., 32)
    max_prediction_length: int = 8 * 4  # prediction horizon (e.g., 8)
    evaluation_horizon_length: int = 2 * 4 # evaluation horizon length (e.g., 2)

    
    validation_percentage: float = 0.1
    test_percentage: float = 0.1
    dataset_cache_dir: str = "/scratch/mohanty/food/ppgr/datasets-cache"
    no_data_cache: bool = False
    
    # Food Covariates    
    include_food_covariates: bool = False
    include_food_covariates_from_horizon: bool = False # If True, the food related covariates are "time varying known reals", else they are "time varying unknown reals"
    
    # User Demographics Covariates
    include_user_demographics_covariates: bool = True
    scale_target_by_user_id: bool = True

    # DataLoader parameters
    batch_size: int = 256
    num_workers: int = 8

    # Model hyperparameters
    
    hidden_size: int = 256
    dropout: float = 0.15
    num_quantiles: int = 7 # 7 or 13 or any odd number > 3
    attention_head_size: int = 4
    hidden_continuous_size: int = 128
    
    share_single_variable_networks: bool = False
    
    variable_selection_network_n_heads: int = 4
        
    transformer_encoder_decoder_num_heads: int = 4
    transformer_encoder_decoder_num_layers: int = 4
    transformer_encoder_decoder_hidden_size: int = 32
    
    enforce_quantile_monotonicity: bool = False

    # Trainer parameters
    max_epochs: int = 30  # Early stopping can likely kick in before this, unless its disabled
    gradient_clip_val: float = 0.1
    
    loss: str = "QuantileLoss" # "QuantileLoss" or "ApproximateCRPS" or "RMSE"    
    optimizer: str = "adamw" # cannot change this atm 
    optimizer_weight_decay: float = 0.05

    lr_scheduler: str = "onecycle" # cannot change this atm 
    
    learning_rate: float = 1e-4
    lr_scheduler_max_lr_multiplier: float = 1.5 # make the lr 1.5 (multiplier) times in the first 10% (pct_start) of the steps, then slowly decrease until the end of all the epochs
    lr_scheduler_pct_start: float = 0.1 # should keep it around upto 2.5 epochs for max 30 epochs - kind of a warm up
    lr_scheduler_anneal_strategy: str = "cos"
    lr_scheduler_cycle_momentum: bool = False
    
    # Precision parameters
    training_precision: str = "32" # "bf16" or "32" (output_layer of the model still runs in fp32)
    
    val_check_interval: float = 0.5
    trainer_log_every_n_steps: int = 1
    number_of_plots_in_metrics_callback: int = 6

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
    
    resume_from_checkpoint: Any = None # path to the checkpoint to resume from

    # Performance parameters
    disable_all_plots: bool = False
    profiler:bool = False


if __name__ == "__main__":
    config = Config()
    rprint("[bold blue]Experiment Configuration:[/bold blue]")
    pprint(config)
    config.quantiles = "asomething"
    pprint(config)
    
    print(asdict(config))