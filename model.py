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


import torch
from torch import nn
import math

from pytorch_forecasting.utils import create_mask

class PPGRTemporalFusionTransformer(TemporalFusionTransformer):
    def __init__(
        self,
        n_head: int = 4,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 8,
        dim_feedforward: int = 512,
        # Add any new hyperparameters you want for your transformer
        # plus the ones from the parent class
        **kwargs,
    ):
        """
        A variant of TemporalFusionTransformer that replaces the LSTM encoder/decoder
        with a PyTorch nn.TransformerEncoder and nn.TransformerDecoder.

        Args:
            n_head (int): number of attention heads in the transformer
            num_encoder_layers (int): number of Transformer encoder layers
            num_decoder_layers (int): number of Transformer decoder layers
            dim_feedforward (int): size of the feed-forward layers in the transformer
            **kwargs: same arguments as TemporalFusionTransformer
        """
        super().__init__(**kwargs)
        
        # Remove the LSTMs
        del self.lstm_encoder
        del self.lstm_decoder

        # Instead, create transformer layers:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.hidden_size,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=self.hparams.dropout,
            batch_first=True,  # If you're using PyTorch 1.9+, we can simplify
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hparams.hidden_size,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        
        # Setup variables needed for the Metrics Callback
        self.validation_batch_full_outputs = []
        self.test_batch_full_outputs = []
        

    def forward(self, x: dict) -> dict:
        """
        Forward pass that swaps out LSTM for a transformer encoder-decoder.
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        timesteps = x_cont.size(1)
        max_encoder_length = int(encoder_lengths.max())

        # embeddings
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        # static variable selection
        if len(self.static_variables) > 0:
            static_embedding = {
                name: input_vectors[name][:, 0] for name in self.static_variables
            }
            static_embedding, static_variable_selection = self.static_variable_selection(
                static_embedding
            )
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

        # encoder variable selection
        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length]
            for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = (
            self.encoder_variable_selection(
                embeddings_varying_encoder,
                static_context_variable_selection[:, :max_encoder_length],
            )
        )

        # decoder variable selection
        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:]
            for name in self.decoder_variables
        }
        embeddings_varying_decoder, decoder_sparse_weights = (
            self.decoder_variable_selection(
                embeddings_varying_decoder,
                static_context_variable_selection[:, max_encoder_length:],
            )
        )

        # --------------------------------------------------------------------
        # Replace LSTM with Transformer
        # --------------------------------------------------------------------
        # We have "embeddings_varying_encoder" for the historical part (B x T_enc x hidden)
        # and "embeddings_varying_decoder" for the future part (B x T_dec x hidden).

        # Construct padding masks for the transformer (True = ignore)
        # We want shape: (B, T_enc) and (B, T_dec)
        encoder_padding_mask = create_mask(
            max_encoder_length, encoder_lengths
        )  # True where "padding"
        decoder_padding_mask = create_mask(
            embeddings_varying_decoder.shape[1], decoder_lengths
        )

        # Optionally create a causal mask for the decoder:
        # Typically, the nn.TransformerDecoder by default uses a causal mask
        # if you pass `tgt_mask`, but you might prefer to let the model attend to
        # all known future steps (the original TFT's "causal_attention" param).
        # For simplicity, let's do standard causal masking:
        T_dec = embeddings_varying_decoder.shape[1]
        causal_mask = nn.Transformer().generate_square_subsequent_mask(T_dec).to(
            embeddings_varying_decoder.device
        )

        # Pass through the encoder
        encoder_output = self.transformer_encoder(
            src=embeddings_varying_encoder,  # B x T_enc x hidden
            src_key_padding_mask=encoder_padding_mask,  # B x T_enc
        )  # -> B x T_enc x hidden

        # Pass through the decoder
        decoder_output = self.transformer_decoder(
            tgt=embeddings_varying_decoder,        # B x T_dec x hidden
            memory=encoder_output,                 # B x T_enc x hidden
            tgt_mask=causal_mask,                  # T_dec x T_dec, standard
            tgt_key_padding_mask=decoder_padding_mask,  # B x T_dec
            memory_key_padding_mask=encoder_padding_mask,  # B x T_enc
        )  # -> B x T_dec x hidden

        # --------------------------------------------------------------------
        # Now mimic the skip connections that were used for LSTM.
        # Instead of post_lstm_gate_encoder/decoder, we have only one decoder output.
        # But let's do something similar:
        # --------------------------------------------------------------------

        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(
            lstm_output_encoder, embeddings_varying_encoder
        )

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(
            lstm_output_decoder, embeddings_varying_decoder
        )

        # Combine them just like in the parent
        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output,
            self.expand_static_context(static_context_enrichment, timesteps),
        )

        # multihead attn over entire sequence
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths
            ),
        )

        attn_output = self.post_attn_gate_norm(
            attn_output, attn_input[:, max_encoder_length:]
        )
        output = self.pos_wise_ff(attn_output)
        output = self.pre_output_gate_norm(
            output, lstm_output[:, max_encoder_length:]
        )

        # Final linear
        if self.n_targets > 1:
            output = [layer(output) for layer in self.output_layer]
        else:
            output = self.output_layer(output)

        # Return in same dictionary format
        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
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
