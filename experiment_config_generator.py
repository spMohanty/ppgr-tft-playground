import yaml
from dataclasses import asdict

from config import Config 

config = Config()

class ExperimentConfigV0:
    """    
    In this set of experiments we will change the following parameters:
    - max_encoder_length: [0.25, 1, 2, 8, 12, 24, 48] (hours)
    - max_prediction_length: [2, 4, 8, 12] (hours)
    - evaluation_horizon_length: [2, 3] (hours)
    - include_food_covariates: [True, False]
    - include_food_covariates_from_horizon: [True, False]
    """
    def __init__(self, config: Config, base_name: str):
        self.config = config
        self.base_name = base_name
        
        self.dataset_version = "v0.4"
        self.wandb_project = "tft-ppgr-2025-ablation"

    def generate_configs(self):
        for max_encoder_length_hr in [0.25, 1, 2, 8, 12, 24, 48]:
            for max_prediction_length_hr in [2, 4, 8, 12]:
                for evaluation_horizon_length_hr in [2, 3]:
                    for include_food_covariates in [True, False]:
                        for include_food_covariates_from_horizon in [True, False]:
                            print(max_encoder_length_hr, max_prediction_length_hr, evaluation_horizon_length_hr, include_food_covariates, include_food_covariates_from_horizon)


if __name__ == "__main__":
    experiment_config = ExperimentConfigV0(config, "tft-ppgr-2025-fo")
    experiment_config.generate_configs()
