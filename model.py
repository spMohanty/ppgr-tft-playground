
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths


import pandas as pd

import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet


class PPGRTemporalFusionTransformer(TemporalFusionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.validation_batch_full_outputs = []
        self.test_batch_full_outputs = []

    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        self.validation_step_outputs.append(log)
        self.validation_batch_full_outputs.append((log, out))
        return log
        
    def test_step(self, batch, batch_idx):
        
        self.test_batch_full_outputs = []        
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        self.testing_step_outputs.append(log)
        self.test_batch_full_outputs.append((log, out))
        
        return log
