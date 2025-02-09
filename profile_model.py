import torch
from model import PPGRTemporalFusionTransformer
from utils import set_random_seeds
from datasets import get_time_series_dataloaders
from config import Config

# Import Rich modules for nice table printing
from rich.console import Console
from rich.table import Table

config = Config()

train_loader, val_loader, test_loader = get_time_series_dataloaders(config)

model = PPGRTemporalFusionTransformer.from_dataset(train_loader.dataset)

print("\n=== Sample Batch from Train Loader ===")
for batch in train_loader:
    past_data, future_data = batch
    print(f"past_data: {past_data}")
    print(f"future_data: {future_data}")
            
    # Use the profiler to capture detailed timings:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU, 
            torch.profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True  # Useful to see where the call is made
    ) as prof:
        # Run the forward pass (or a batch of them)
        out = model(past_data)
    break

# Dynamically extract keys from the profile events and build a Rich table.
console = Console()

# Retrieve profiler events (no sorting here, but you could sort if needed)
profile_events = prof.key_averages()

# If we have events, extract the keys from the first event.
if profile_events:
    event_keys = list(vars(profile_events[0]).keys())
else:
    event_keys = []

rich_table = Table(title="Profiler Summary (Extracted Keys)")

# Create columns from the event keys.
for key in event_keys:
    rich_table.add_column(key, style="cyan", no_wrap=True)

# Add rows for each profiler event.
for ev in profile_events:
    row = []
    for key in event_keys:
        value = getattr(ev, key)
        # If the key name suggests a time measurement and is numeric,
        # convert microseconds to milliseconds.
        if isinstance(value, (int, float)) and "time" in key:
            # Assume the value is in microseconds from the profiler and convert to ms.
            value = f"{value / 1000.0:.3f}"
        else:
            value = str(value)
        row.append(value)
    rich_table.add_row(*row)

console.print(rich_table)
prof.export_chrome_trace("trace.json")
    
    
