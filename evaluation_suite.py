#!/usr/bin/env python3

from config import Config
from datasets import get_time_series_dataloaders
from model import PPGRTemporalFusionTransformer
import torch

def move_to_device(data, device):
    """
    Recursively moves tensors in a nested structure (dict, list, tuple) to the given device.
    """
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data

def main():
    checkpoint_path = "./tft-ppgr-2025-debug/h5ofur8j/checkpoints/epoch=0-step=1.ckpt"
    config = Config(
        resume_from_checkpoint=checkpoint_path
    )    
    
    # Load data with pin_memory for faster transfers
    train_loader, val_loader, test_loader = get_time_series_dataloaders(config)
    
    # Load model
    model = PPGRTemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    
    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Run prediction for the first batch in the test set
        for batch in train_loader:
            # Efficiently move the entire nested batch to the chosen device
            past_data, future_data = move_to_device(batch, device)
            predictions = model.forward(past_data)
            break
    
    # # Get predictions
    # predictions = model.predict(batch)
    # print(predictions)
    # breakpoint()
    
    
    pass


if __name__ == "__main__":
    main()