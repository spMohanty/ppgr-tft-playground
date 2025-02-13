#!/usr/bin/env python3

from config import Config
from datasets import get_time_series_dataloaders
from model import PPGRTemporalFusionTransformer

def main():
    
    checkpoint_path = "./tft-ppgr-2025-debug/s8iqcn79/checkpoints/epoch=29-step=3269.ckpt"
    # config = Config(
    #     resume_from_checkpoint=checkpoint_path
    # )    
    
    # # Load data
    # train_loader, val_loader, test_loader = get_time_series_dataloaders(config)
    
    # Load model
    model = PPGRTemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    
    # model.eval()
    
    # # Run prediction for the first batch in the test set
    # batch = next(iter(test_loader))
    # print(batch)
    
    # # Get predictions
    # predictions = model.predict(batch)
    # print(predictions)
    # breakpoint()
    
    
    pass


if __name__ == "__main__":
    main()