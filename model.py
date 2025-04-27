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

from dataclasses import asdict

# Third-party imports
import numpy as np
import pandas as pd  # Only one import is needed
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from loguru import logger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet

from typing import Any, Dict, Optional, Union, List

import torch
from torch import nn
import math

from pytorch_forecasting.utils import create_mask

from pytorch_forecasting.metrics.base_metrics import Metric
from pytorch_forecasting.metrics import (
    MAE,
    MASE,
    SMAPE,
    DistributionLoss,
    MultiHorizonMetric,
    MultiLoss,
    QuantileLoss,
    convert_torchmetric_to_pytorch_forecasting_metric,
)

from pytorch_forecasting.utils import to_list

from sub_modules import (
    SharedTransformerEncoder,
    SharedTransformerDecoder,
    GatedTransformerLSTMProjectionUnit,
    TransformerVariableSelectionNetwork,
    PreNormResidualBlock
)

from sub_modules import GatedLinearUnit, AddNorm, GateAddNorm


from utils import conditional_enforce_quantile_monotonicity

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) as proposed in https://arxiv.org/abs/2104.09864.
    
    This implementation supports offset indices to handle negative positions.
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        offset: int = 0
    ) -> None:
        """
        Initialize rotary positional embeddings.
        
        Args:
            dim: Hidden dimension size. Must be divisible by 2.
            max_seq_len: Maximum sequence length the model is expected to handle.
                         This determines the size of the position cache.
            base: Base value for the frequency calculations.
            offset: Position offset to handle negative indices (centering).
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.offset = offset
        
        assert dim % 2 == 0, "RoPE dim must be divisible by 2"
        self.rope_init()

    def rope_init(self):
        # Precompute frequency bands
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes from 0 to max_seq_len-1
        seq_idx = torch.arange(
            max_seq_len, dtype=torch.float, device=self.theta.device
        )

        # Outer product of theta and position index
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta)
        
        # Cache includes both the cos and sin components
        # Shape: [max_seq_len, dim//2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            positions: Position indices of shape [batch_size, seq_len]
            
        Returns:
            Tensor with rotary embeddings applied
        """
        # Ensure positions include offset (for negative positions handling)
        positions = positions + self.offset
        
        # Ensure position indices are within bounds
        positions = torch.clamp(positions, 0, self.max_seq_len - 1)
        
        # Get batch size and sequence length
        batch_size, seq_len = positions.shape
        
        # Get cached embeddings for the given positions
        # Shape after indexing: [batch_size, seq_len, dim//2, 2]
        rope_cache = self.cache[positions]
        
        # Reshape input for rotation
        # From [batch_size, seq_len, dim] to [batch_size, seq_len, dim//2, 2]
        x_reshaped = x.float().reshape(batch_size, seq_len, -1, 2)
        
        # Apply rotary transformation using the cached values
        # cos(θ)x - sin(θ)y and sin(θ)x + cos(θ)y
        x_out = torch.stack(
            [
                x_reshaped[..., 0] * rope_cache[..., 0] - 
                x_reshaped[..., 1] * rope_cache[..., 1],
                
                x_reshaped[..., 1] * rope_cache[..., 0] + 
                x_reshaped[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        
        # Reshape back to original shape - ensure this matches the input shape
        x_out = x_out.reshape(batch_size, seq_len, self.dim)
        
        # Return with the same dtype as the input
        return x_out.type_as(x)

class PPGRTemporalFusionTransformer(TemporalFusionTransformer):
    def __init__(
        self,        
        # Model architecture
        hidden_size: int, # for the main model flow
        hidden_continuous_size: int = 1, # for the main model flow
        attention_head_size: int = 1, # interpretable attention head 

        
        output_size: Union[int, List[int]] = 7, # number of quantiles in this case
        dropout: float = 0.1, 
        
        # Optimizer and learning rate scheduler settings
        optimizer: str = "adamw",
        lr_scheduler: str = "onecycle",
        lr_scheduler_max_lr_multiplier: float = 1.0,
        lr_scheduler_pct_start: float = 0.3,
        lr_scheduler_anneal_strategy: str = "cosine",
        lr_scheduler_cycle_momentum: bool = True,
        learning_rate: float = 1e-3,
        optimizer_weight_decay: float = 0.0,

        # Variable Selection Networks
        variable_selection_network_n_heads: int = 4,
        share_single_variable_networks: bool = True,
                
        # Transformer encoder/decoder configuration
        transformer_encoder_decoder_num_heads: int = 8,
        transformer_encoder_decoder_hidden_size: int = 32,
        transformer_encoder_decoder_num_layers: int = 1,        
        
        # Additional inputs and settings
        max_encoder_length: int = 8 * 4, # 8 hours in 15 min steps
        enforce_quantile_monotonicity: bool = False,
        
        # Positional embeddings
        use_rotary_positional_embeddings: bool = False,
        
        **kwargs,
    ):
        """
        A variant of TemporalFusionTransformer that replaces the LSTM encoder/decoder
        with a PyTorch nn.TransformerEncoder and nn.TransformerDecoder.

        Args:
            experiment_config (Config): configuration for the experiment (check config.py)
            **kwargs: same arguments as TemporalFusionTransformer
        """        
        # Setup variables needed for the Metrics Callback
        self.save_hyperparameters()

        super().__init__(**kwargs)        
        
        # Initialize rotary positional embeddings if enabled
        self.use_rotary_positional_embeddings = use_rotary_positional_embeddings
        if self.use_rotary_positional_embeddings:
            logger.info("Initializing rotary positional embeddings")
            total_range = max_encoder_length + 100  # a safe upper bound
            self.positional_embeddings = RotaryPositionalEmbeddings(
                dim=hidden_size,
                base=10000,
                offset=max_encoder_length  # Center positions around the last valid position
            )
        
        logger.info("Setting up variable selection networks")
        self.setup_variable_selection_networks()
        
        logger.info("Setting up transformer encoder decoder layers")
        self.setup_transformer_encoder_decoder_layers()
    
        logger.info("Cleaning up LSTM encoder decoder layers")
        self.clean_up_lstm_encoder_decoder_layers()
        
        logger.info("Setting up static context encoders")
        self.setup_static_context_encoders()
        
        logger.info("Setting up output layers")
        self.setup_output_layers()
    
    def setup_static_context_encoders(self):
        ## Static Encoders
        # for variable selection
        self.static_context_variable_selection = PreNormResidualBlock(
            input_dim=self.hparams.hidden_size,
            hidden_dim=self.hparams.hidden_size,
            output_dim=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_dim=None,
        )
        
        # for post lstm static enrichment
        self.static_context_enrichment = PreNormResidualBlock(
            input_dim=self.hparams.hidden_size,
            hidden_dim=self.hparams.hidden_size,
            output_dim=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_dim=None,
        )
        
        # static enrichment just before the multihead attn
        self.static_enrichment = PreNormResidualBlock(
            input_dim=self.hparams.hidden_size,
            hidden_dim=self.hparams.hidden_size,
            output_dim=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_dim=self.hparams.hidden_size,
        )
    
    def setup_output_layers(self):
        # post multihead attn processing before the output processing
        self.pos_wise_ff = PreNormResidualBlock(
            input_dim=self.hparams.hidden_size,
            hidden_dim=self.hparams.hidden_size,
            output_dim=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_dim=None,
        )
        
        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(
            self.hparams.hidden_size, dropout=0.0, trainable_add=False
        )
        if self.n_targets > 1:  # if to run with multiple targets
            self.output_layer = nn.ModuleList(
                [
                    nn.Linear(self.hparams.hidden_size, output_size)
                    for output_size in self.hparams.output_size
                ]
            )
        else:
            self.output_layer = nn.Linear(
                self.hparams.hidden_size, self.hparams.output_size
            )
        
            
    
    def configure_optimizers(self):
        assert self.hparams.optimizer == "adamw", "Only adamw optimizer is supported atm"
        assert self.hparams.lr_scheduler in ["onecycle", "none"], "Only onecycle scheduler is supported atm or none or auto"
        
        assert self.hparams.lr_scheduler_max_lr_multiplier >= 1.0, "lr_scheduler_max_lr_multiplier must be >= 1.0"
        
        optimizer = AdamW(self.parameters(), 
                          lr=self.hparams.learning_rate, 
                          weight_decay=self.hparams.optimizer_weight_decay)

        # estimate total number of training steps
        total_steps = self.trainer.estimated_stepping_batches
        
        lr_scheduler = {}
        if self.hparams.lr_scheduler == "onecycle":
            # Configure the OneCycleLR scheduler
            lr_scheduler = {
                'scheduler': OneCycleLR(
                    optimizer,
                    max_lr=self.hparams.learning_rate * self.hparams.lr_scheduler_max_lr_multiplier,           # Peak learning rate during the cycle
                    total_steps=total_steps,  # Total number of training steps
                pct_start=self.hparams.lr_scheduler_pct_start,         # Fraction of steps spent increasing the LR
                anneal_strategy=self.hparams.lr_scheduler_anneal_strategy, # Cosine annealing for LR decay
                cycle_momentum=self.hparams.lr_scheduler_cycle_momentum   # Set to True if you wish to cycle momentum
            ),
            'interval': 'step',        # Update the scheduler every training step
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }
        
    def clean_up_lstm_encoder_decoder_layers(self):
        self.static_context_initial_hidden_lstm = None
        self.static_context_initial_cell_lstm = None
        self.lstm_encoder = None
        self.lstm_decoder = None
        self.post_lstm_gate_encoder = None
        self.post_lstm_gate_decoder = None
        self.post_lstm_add_norm_encoder = None
        self.post_lstm_add_norm_decoder = None
        # empty cuda cache
        torch.cuda.empty_cache() # called only once at the beginning, so not that big of a deal
        
    def setup_transformer_encoder_decoder_layers(self):
        # Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.hidden_size,
            nhead=self.hparams.transformer_encoder_decoder_num_heads,
            dim_feedforward=self.hparams.transformer_encoder_decoder_hidden_size,
            dropout=self.hparams.dropout,
            batch_first=True,  
        )
        self.transformer_encoder = SharedTransformerEncoder(
            layer=encoder_layer,
            num_layers=self.hparams.transformer_encoder_decoder_num_layers,
        )

        # Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hparams.hidden_size,
            nhead=self.hparams.transformer_encoder_decoder_num_heads,
            dim_feedforward=self.hparams.transformer_encoder_decoder_hidden_size,
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        self.transformer_decoder = SharedTransformerDecoder(
            layer=decoder_layer,
            num_layers=self.hparams.transformer_encoder_decoder_num_layers,
        )
        
        # skip connection for transformer encoder and decoder
        self.post_transformer_gate_encoder = GatedLinearUnit(
            self.hparams.hidden_size, dropout=self.hparams.dropout
        )
        self.post_transformer_gate_decoder = self.post_transformer_gate_encoder
        self.post_transformer_add_norm_encoder = AddNorm(
            self.hparams.hidden_size, trainable_add=False
        )
        self.post_transformer_add_norm_decoder = self.post_transformer_add_norm_encoder
        

    def setup_variable_selection_networks(self):
        # variable selection
        # variable selection for static variables
        static_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.hparams.static_categoricals
        }
        static_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.static_reals
            }
        )
                
        self.static_variable_selection = TransformerVariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hparams.hidden_size,
            n_heads=self.hparams.variable_selection_network_n_heads,
            input_embedding_flags={
                name: True for name in self.hparams.static_categoricals
            },
            dropout=self.hparams.dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.time_varying_reals_encoder
            }
        )

        decoder_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.time_varying_reals_decoder
            }
        )

        # create single variable grns that are shared across decoder and encoder
        if self.hparams.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = PreNormResidualBlock(
                    input_dim=input_size,
                    hidden_dim=min(input_size, self.hparams.hidden_size),
                    output_dim=self.hparams.hidden_size,
                    dropout=self.hparams.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = PreNormResidualBlock(
                        input_dim=input_size,
                        hidden_dim=min(input_size, self.hparams.hidden_size),
                        output_dim=self.hparams.hidden_size,
                        dropout=self.hparams.dropout,
                    )


        self.encoder_variable_selection = TransformerVariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            n_heads=self.hparams.variable_selection_network_n_heads,
            input_embedding_flags={
                name: True for name in self.hparams.time_varying_categoricals_encoder
            },
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not self.hparams.share_single_variable_networks
                else self.shared_single_variable_grns
            ),
        )

        self.decoder_variable_selection = TransformerVariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            n_heads=self.hparams.variable_selection_network_n_heads,
            input_embedding_flags={
                name: True for name in self.hparams.time_varying_categoricals_decoder
            },
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not self.hparams.share_single_variable_networks
                else self.shared_single_variable_grns
            ),
        )
        
    def forward(self, x: dict) -> dict:
        """
        Forward pass that swaps out LSTM for a transformer encoder-decoder.
        """        
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        timesteps = x_cont.size(1)
        max_encoder_length = self.hparams.max_encoder_length

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
        if embeddings_varying_encoder.ndim == 2:
            # If the time dimension got squeezed out, unsqueeze it.
            embeddings_varying_encoder = embeddings_varying_encoder.unsqueeze(1)
        

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
        if embeddings_varying_decoder.ndim == 2:
            embeddings_varying_decoder = embeddings_varying_decoder.unsqueeze(1)

        
        # --------------------------------------------------------------------
        # Apply positional encodings if enabled
        # --------------------------------------------------------------------
        if self.use_rotary_positional_embeddings:
            # Generate position indices for encoder and decoder
            B = embeddings_varying_encoder.size(0)
            T_past = embeddings_varying_encoder.size(1)
            T_future = embeddings_varying_decoder.size(1)
            device = embeddings_varying_encoder.device
            
            # Generate time indices for positional embeddings
            past_indices = torch.arange(T_past, device=device).unsqueeze(0).expand(B, -1)
            future_indices = torch.arange(T_future, device=device).unsqueeze(0).expand(B, -1) + T_past
            
            # Center indices to make the last valid position have index 0
            offset_value = max_encoder_length - 1
            centered_offset = offset_value * torch.ones((B, 1), device=device, dtype=torch.long)
            
            # Apply centering to indices
            past_indices = past_indices - centered_offset
            future_indices = future_indices - centered_offset
            
            # Apply rotary positional embeddings
            embeddings_varying_encoder = self.positional_embeddings(
                embeddings_varying_encoder, past_indices
            )
            
            embeddings_varying_decoder = self.positional_embeddings(
                embeddings_varying_decoder, future_indices
            )

        # --------------------------------------------------------------------
        # Process inputs with Transformer
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

        # create a causal mask for the decoder:
        # Typically, the nn.TransformerDecoder by default uses a causal mask
        # if you pass `tgt_mask`, but you might prefer to let the model attend to
        # all known future steps (the original TFT's "causal_attention" param).
        # For simplicity, let's do standard causal masking:
        T_dec = embeddings_varying_decoder.shape[1]
        causal_mask = nn.Transformer().generate_square_subsequent_mask(T_dec).to(
            embeddings_varying_decoder.device
        )
        
        # Pass through the encoder
        transformer_encoder_output = self.transformer_encoder(
            src=embeddings_varying_encoder,  # B x T_enc x hidden
            src_key_padding_mask=encoder_padding_mask,  # B x T_enc
        )  # -> B x T_enc x hidden

        # Pass through the decoder
        transformer_decoder_output = self.transformer_decoder(
            tgt=embeddings_varying_decoder,        # B x T_dec x hidden
            memory=transformer_encoder_output,                 # B x T_enc x hidden
            tgt_mask=causal_mask,                  # T_dec x T_dec, standard
            tgt_key_padding_mask=decoder_padding_mask,  # B x T_dec
            memory_key_padding_mask=encoder_padding_mask,  # B x T_enc
        )  # -> B x T_dec x hidden


        # Add post transformer gating
        transformer_output_encoder = self.post_transformer_gate_encoder(transformer_encoder_output)
        transformer_output_encoder = self.post_transformer_add_norm_encoder(
            transformer_output_encoder, embeddings_varying_encoder
        )

        transformer_output_decoder = self.post_transformer_gate_decoder(transformer_decoder_output)
        transformer_output_decoder = self.post_transformer_add_norm_decoder(
            transformer_output_decoder, embeddings_varying_decoder
        )

        transformer_output = torch.cat([transformer_output_encoder, transformer_output_decoder], dim=1)

                
        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(transformer_output,
                                            self.expand_static_context(
                                                static_context_enrichment, 
                                                timesteps))

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
            output, transformer_output[:, max_encoder_length:]
        )
        
        # Ensure the final output layer always runs in full precision
        with torch.amp.autocast("cuda", enabled=False):
            # Final linear        
            if self.n_targets > 1:
                output = []
                for layer in self.output_layer:
                    output.append(conditional_enforce_quantile_monotonicity(layer(output), self.hparams.enforce_quantile_monotonicity))
            else:
                output = conditional_enforce_quantile_monotonicity(self.output_layer(output), self.hparams.enforce_quantile_monotonicity)


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
        
    def training_step(self, batch, batch_idx):
        """
        Training step: processes a training batch, updates logging, and stores outputs.
        """
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        self.training_step_outputs.append(log) # from the base_model.py 
        self.last_training_batch_output = (log, out)
        
        # Access and log the current learning rate from the first optimizer's first param group
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_epoch=True, prog_bar=True, logger=True)
        
        
        return log

    def validation_step(self, batch, batch_idx):
        """
        Validation step: processes a validation batch, updates logging, and stores outputs.
        """
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        self.validation_step_outputs.append(log)
        self.last_validation_batch_output = (log, out)
        return log

    def test_step(self, batch, batch_idx):
        """
        Test step: processes a test batch, updates logging, and stores outputs.
        """
        # Clear previous test outputs.
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        self.testing_step_outputs.append(log)
        self.last_test_batch_output = (log, out)
        return log

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int = 0,
        add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
        show_future_observed: bool = True,
        ax=None,
        quantiles_kwargs: Optional[Dict[str, Any]] = None,
        prediction_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Plot prediction of prediction vs actuals

        Args:
            x: network input
            out: network output
            idx: index of prediction to plot
            add_loss_to_title: if to add loss to title or loss function to calculate. Can be either metrics,
                bool indicating if to use loss metric or tensor which contains losses for all samples.
                Calcualted losses are determined without weights. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on
            quantiles_kwargs (Dict[str, Any]): parameters for ``to_quantiles()`` of the loss metric.
            prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.

        Returns:
            matplotlib figure
        """  # noqa: E501
        if quantiles_kwargs is None:
            quantiles_kwargs = {}
        if prediction_kwargs is None:
            prediction_kwargs = {}

        from matplotlib import pyplot as plt
        import seaborn as sns

        # Set the seaborn style and palette
        sns.set_style("white")  # Changed from "whitegrid" to "white"
        sns.set_palette("tab10")
        palette = sns.color_palette("tab10")
        
        # Get colors from the palette
        historical_color = palette[0]  # blue
        ground_truth_color = palette[1]  # orange
        quantile_fill_color = historical_color 
        forecast_color = "darkblue" 
        grid_color = sns.color_palette("Greys")[2]
        meal_color = "purple"

        # all true values for y of the first sample in batch
        encoder_targets = to_list(x["encoder_target"])
        decoder_targets = to_list(x["decoder_target"])
        
        encoder_continuous = to_list(x["encoder_cont"])
        decoder_continuous = to_list(x["decoder_cont"])
        
        
        
        food_params_as_unknown_reals = self.hparams.dataset_parameters["time_varying_unknown_reals"]
        food_params_as_unknown_reals = [param for param in food_params_as_unknown_reals if param.startswith("food__")]
        

        y_raws = to_list(
            out["prediction"]
        )  # raw predictions - used for calculating loss
        y_hats = to_list(self.to_prediction(out, **prediction_kwargs))
        y_quantiles = to_list(self.to_quantiles(out, **quantiles_kwargs))

        # for each target, plot
        figs = []
        for y_raw, y_hat, y_quantile, encoder_target, decoder_target, encoder_continuous, decoder_continuous in zip(
            y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets, encoder_continuous, decoder_continuous
        ):
            y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
            max_encoder_length = x["encoder_lengths"].max()
            y = torch.cat(
                (
                    y_all[: x["encoder_lengths"][idx]],
                    y_all[
                        max_encoder_length : (
                            max_encoder_length + x["decoder_lengths"][idx]
                        )
                    ],
                ),
            )
            # move predictions to cpu
            y_hat = y_hat.detach().cpu().to(torch.float32)[idx, : x["decoder_lengths"][idx]]
            y_quantile = y_quantile.detach().cpu().to(torch.float32)[idx, : x["decoder_lengths"][idx]]
            y_raw = y_raw.detach().cpu().to(torch.float32)[idx, : x["decoder_lengths"][idx]]

            # move to cpu
            y = y.detach().cpu().to(torch.float32)
            
            # create figure
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            else:
                fig = ax.get_figure()
                
            n_pred = y_hat.shape[0]
            
            # Create time indices for both history and forecast
            T_context = y.shape[0] - n_pred
            T_forecast = n_pred
            x_hist = list(range(-T_context + 1, 1))
            x_pred = list(range(1, T_forecast + 1))
            
            # Plot historical glucose - check if we need scatter or plot
            if len(x_hist) > 1:
                ax.plot(x_hist, y[:-n_pred], marker="o", markersize=3, 
                       label="Historical Glucose", color=historical_color, linewidth=1.5)
            else:
                ax.scatter(x_hist, y[:-n_pred], marker="o", s=30, 
                          label="Historical Glucose", color=historical_color)
            
            # Plot ground truth (if requested)
            if show_future_observed:
                if len(x_pred) > 1:
                    ax.plot(x_pred, y[-n_pred:], marker="o", markersize=3,
                           label="Ground Truth", color=ground_truth_color, linewidth=1.5)
                else:
                    ax.scatter(x_pred, y[-n_pred:], marker="o", s=30,
                              label="Ground Truth", color=ground_truth_color)
            
            # Plot median forecast
            median_index = y_quantile.shape[1] // 2
            if len(x_pred) > 1:
                ax.plot(x_pred, y_hat, marker="o", markersize=3, 
                       label="Median Forecast", color=forecast_color, linewidth=1.5)
            else:
                ax.scatter(x_pred, y_hat, marker="o", s=30, 
                          label="Median Forecast", color=forecast_color)
            
            # Plot quantile intervals with gradient opacity, similar to plot_forecast_examples
            # The median_index is the middle point (e.g., 4 if there are 9 quantiles)
            for qi in range(y_quantile.shape[1] - 1):
                # Calculate alpha based on distance from median, matching plot_forecast_examples approach
                alpha_val = 0.1 + (abs(qi - median_index)) * 0.05
                
                # Get lower and upper bounds (ensuring they're in correct order)
                q_lower = torch.minimum(y_quantile[:, qi], y_quantile[:, qi+1])
                q_upper = torch.maximum(y_quantile[:, qi], y_quantile[:, qi+1])
                
                if len(x_pred) > 1:
                    ax.fill_between(
                        x_pred,
                        q_lower,
                        q_upper,
                        color=quantile_fill_color,
                        alpha=alpha_val
                    )
                else:
                    # Handle single point case
                    quantiles = torch.tensor(
                        [[y_quantile[0, qi]], [y_quantile[0, qi+1]]]
                    )
                    ax.errorbar(
                        x_pred,
                        y[[-n_pred]],
                        yerr=quantiles - y[-n_pred],
                        color=quantile_fill_color,
                        capsize=1.0,
                    )

            # Add food consumption lines
            food_key = "food__food_intake_row"
            if food_key in self.hparams.dataset_parameters["scalers"].keys():
                food_intake_row_scaler = self.hparams.dataset_parameters["scalers"][food_key]
                is_food_in_future = len(food_params_as_unknown_reals) > 0
                
                index_food_eaten = self.hparams.x_reals.index(food_key)
                
                # Get unscaled values 
                food_intake_row_scaled_past = encoder_continuous[idx, :, index_food_eaten].detach().cpu().numpy()
                if is_food_in_future: food_intake_row_scaled_future  = decoder_continuous[idx, :, index_food_eaten].detach().cpu().numpy()

                # Better approach to detect boolean values in scaled data
                # First check if we only have a single item or multiple items
                if len(food_intake_row_scaled_past) == 1:
                    # When there's only one item and data is food-anchored, it's always True
                    food_intake_row_past = np.ones_like(food_intake_row_scaled_past, dtype=bool)
                else:
                    # When we have multiple items, we can use the approach of comparing against min
                    food_intake_row_past = food_intake_row_scaled_past > food_intake_row_scaled_past.min()
                
                if is_food_in_future:
                    if len(food_intake_row_scaled_future) == 1:
                        # When there's only one item and data is food-anchored, it's always True
                        food_intake_row_future = np.ones_like(food_intake_row_scaled_future, dtype=bool)
                    else:
                        food_intake_row_future = food_intake_row_scaled_future > food_intake_row_scaled_future.min()

                food_intake_row_indices_past = np.where(food_intake_row_past)[0]
                if is_food_in_future: food_intake_row_indices_future = np.where(food_intake_row_future)[0]
                
                # update indices to the relative timesteps
                food_intake_row_indices_past = food_intake_row_indices_past - T_context + 1
                if is_food_in_future: food_intake_row_indices_future = food_intake_row_indices_future  + 1
                
                
                try:
                    # Double check to ensure that there is a food intake at relative timestep 0
                    assert 0 in food_intake_row_indices_past, "There should be a food intake at the last timestep of the past indices"
                except Exception as e:
                    breakpoint()
            
                # Plot vertical lines for each meal consumption time
                meal_label_added = False
                
                meal_consumption_indices = food_intake_row_indices_past
                
                if is_food_in_future:
                    meal_consumption_indices = np.concatenate([food_intake_row_indices_past, food_intake_row_indices_future])
                
                for idx in meal_consumption_indices:
                    ax.axvline(x=idx, color=meal_color, linestyle="--", alpha=0.7, 
                              label="Meal Consumption" if not meal_label_added else None)
                    meal_label_added = True
                                
    
            if add_loss_to_title is not False:
                if isinstance(add_loss_to_title, bool):
                    loss = self.loss
                elif isinstance(add_loss_to_title, torch.Tensor):
                    loss = add_loss_to_title.detach()[idx].item()
                elif isinstance(add_loss_to_title, Metric):
                    loss = add_loss_to_title
                else:
                    raise ValueError(
                        f"add_loss_to_title '{add_loss_to_title}'' is unknown"
                    )
                if isinstance(loss, MASE):
                    loss_value = loss(
                        y_raw[None], (y[-n_pred:][None], None), y[:n_pred][None]
                    )
                elif isinstance(loss, Metric):
                    try:
                        loss_value = loss(y_raw[None], (y[-n_pred:][None], None))
                    except Exception:
                        loss_value = "-"
                else:
                    loss_value = loss
                ax.set_title(f"Glucose Forecast - #{idx} (Loss: {loss_value})", fontsize=13, fontweight='bold')
            else:
                ax.set_title(f"Glucose Forecast - #{idx}", fontsize=13, fontweight='bold')
                
            # Add proper labels
            ax.set_xlabel("Relative Timestep", fontsize=11)
            ax.set_ylabel("Glucose Level (mmol/L)", fontsize=11)
            
            # Remove grid lines - we're using a clean white background now
            ax.grid(False)
            
            # Add a vertical line at t=0 to mark the forecast start
            # ax.axvline(x=0, color=palette[4], linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Improve legend
            ax.legend(loc="best", fontsize=10, framealpha=0.9, edgecolor=grid_color)
            
            # Use pure white background
            ax.set_facecolor('white')
            
            # Add subtle spines
            for spine in ax.spines.values():
                spine.set_edgecolor(grid_color)
                spine.set_linewidth(0.8)
            
            # Adjust layout
            fig.tight_layout()
            
            figs.append(fig)

        # return multiple of target is a list, otherwise return single figure
        if isinstance(x["encoder_target"], (tuple, list)):
            return figs
        else:
            return fig