#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module: ppgr_tft
Description: Custom implementation of the Temporal Fusion Transformer (TFT) with extensions.
             The PPGRTemporalFusionTransformer class extends the pytorch_forecasting
             TemporalFusionTransformer and adds custom behavior for validation, testing,
             and (in the future) post-training modes.
"""

import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# Standard library imports
from pathlib import Path
import copy
import math

# Third-party imports
import numpy as np
import pandas as pd  # Only one import is needed
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet


class PPGRTemporalFusionTransformer(TemporalFusionTransformer):
    """
    Custom Temporal Fusion Transformer with additional functionalities.
    
    This model extends the pytorch_forecasting TemporalFusionTransformer by
    providing custom validation and test step handling, as well as a hook for a
    post-training mode (currently not implemented).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Containers to store full outputs for validation and test steps.
        self.validation_batch_full_outputs = []
        self.test_batch_full_outputs = []
        
        # Flag for post-training mode. (Not implemented in this version)
        self.post_train_mode = False
        
        # Uncomment and define your custom attention layer if needed:
        # self.multihead_attn = PPGRVectorizedInterpretableMultiHeadAttention(
        #     d_model=self.hparams.hidden_size,
        #     n_head=self.hparams.attention_head_size,
        #     dropout=self.hparams.dropout,
        # )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        If post_train_mode is False, it delegates to the parent class's forward method.
        Otherwise, it is expected to use a custom output layer (not implemented).
        """
        # Get backbone outputs
        (
            backbone_output,
            attn_output_weights,
            max_encoder_length,
            static_variable_selection,
            encoder_sparse_weights,
            decoder_sparse_weights,
            decoder_lengths,
            encoder_lengths,
        ) = self._forward_backbone(x)
    
        if not self.post_train_mode:
            # Use the standard output layer provided by the parent class.
            return super().forward(x)
        else:
            # Custom post-training mode logic goes here (not yet implemented).
            raise NotImplementedError("Post-train mode not implemented")

    def _forward_backbone(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Processes the input through embeddings, LSTM encoder/decoder, static enrichment,
        and attention mechanisms to produce the backbone representation.
        
        Args:
            x (Dict[str, torch.Tensor]): Dictionary containing model inputs.
            
        Returns:
            Tuple containing:
                - backbone_output: The output from the temporal fusion decoder.
                - attn_output_weights: Attention weights from the multihead attention.
                - max_encoder_length: Maximum length of the encoder inputs.
                - static_variable_selection: Weights from static variable selection.
                - encoder_sparse_weights: Weights from encoder variable selection.
                - decoder_sparse_weights: Weights from decoder variable selection.
                - decoder_lengths: Lengths of the decoder sequences.
                - encoder_lengths: Lengths of the encoder sequences.
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        
        # Concatenate encoder and decoder categorical and continuous features along time dimension.
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        timesteps = x_cont.size(1)  # Total time steps (encoder + decoder)
        max_encoder_length = int(encoder_lengths.max())
        
        # Create input embeddings for categorical features.
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update({
            name: x_cont[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.hparams.x_reals)
            if name in self.reals
        })

        # Static embeddings and variable selection.
        if len(self.static_variables) > 0:
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            static_variable_selection = torch.zeros(
                (x_cont.size(0), 0), dtype=self.dtype, device=self.device
            )

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        # Process encoder and decoder variables separately.
        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length]
            for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:]
            for name in self.decoder_variables
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # Initialize LSTM states using static context.
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )

        # Run the encoder LSTM.
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder,
            (input_hidden, input_cell),
            lengths=encoder_lengths,
            enforce_sorted=False,
        )

        # Run the decoder LSTM.
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # Apply skip connections over LSTM outputs.
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)
        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)
        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # Static enrichment.
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output,
            self.expand_static_context(static_context_enrichment, timesteps),
        )

        # Attention mechanism.
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # Query only for prediction time steps.
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths),
        )

        # Apply skip connection over attention output.
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])
        output = self.pos_wise_ff(attn_output)

        # Final skip connection before the output layer.
        backbone_output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])

        return (
            backbone_output,
            attn_output_weights,
            max_encoder_length,
            static_variable_selection,
            encoder_sparse_weights,
            decoder_sparse_weights,
            decoder_lengths,
            encoder_lengths,
        )

    def validation_step(self, batch, batch_idx):
        """
        Validation step: processes a validation batch, updates logging, and stores outputs.
        """
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        self.validation_step_outputs.append(log)
        self.validation_batch_full_outputs.append((log, out))
        return log

    def test_step(self, batch, batch_idx):
        """
        Test step: processes a test batch, updates logging, and stores outputs.
        """
        # Clear previous test outputs.
        self.test_batch_full_outputs = []        
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        self.testing_step_outputs.append(log)
        self.test_batch_full_outputs.append((log, out))
        return log
