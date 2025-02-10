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

from typing import Any, Dict, Optional, Union

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
    PatchedVariableSelectionNetwork
)

from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import GatedResidualNetwork

from utils import conditional_enforce_quantile_monotonicity

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from config import Config


class PPGRTemporalFusionTransformer(TemporalFusionTransformer):
    def __init__(
        self,
        transformer_encoder_decoder_num_heads: int = 4,
        transformer_encoder_decoder_num_layers: int = 4,
        transformer_encoder_decoder_hidden_size: int = 32,
        
        use_transformer_encoder_decoder_layers: bool = True,
        use_transformer_variable_selection_networks: bool = False,
        
        enforce_quantile_monotonicity: bool = False,
        
        experiment_config: Config = None,

        **kwargs,
    ):
        """
        A variant of TemporalFusionTransformer that replaces the LSTM encoder/decoder
        with a PyTorch nn.TransformerEncoder and nn.TransformerDecoder.

        Args:
            transformer_encoder_decoder_num_heads (int): number of attention heads in the transformer
            transformer_encoder_decoder_num_layers (int): number of Transformer encoder layers
            transformer_encoder_decoder_hidden_size (int): size of the feed-forward layers in the transformer
            **kwargs: same arguments as TemporalFusionTransformer
        """
        super().__init__(**kwargs)
        
        # Setup custom hyperparameters
        # Transformer encoder-decoder layers
        # self.hparams.use_transformer_encoder_decoder_layers = use_transformer_encoder_decoder_layers
        # self.hparams.transformer_encoder_decoder_num_heads = transformer_encoder_decoder_num_heads
        # self.hparams.transformer_encoder_decoder_num_layers = transformer_encoder_decoder_num_layers
        # self.hparams.transformer_encoder_decoder_hidden_size = transformer_encoder_decoder_hidden_size
        
        # # Variable selection networks
        # self.hparams.use_transformer_variable_selection_networks = use_transformer_variable_selection_networks
        
        # self.hparams.enforce_quantile_monotonicity = enforce_quantile_monotonicity
        
        self.hparams.update(asdict(experiment_config))
        
                
        # Setup variables needed for the Metrics Callback
        self.training_batch_full_outputs = []
        self.validation_batch_full_outputs = []
        self.test_batch_full_outputs = []
        
        self.transformer_encoder_decoder_num_layers = transformer_encoder_decoder_num_layers
        
        logger.info("Setting up variable selection networks")
        self.setup_variable_selection_networks()
        
        if self.hparams.use_transformer_encoder_decoder_layers:
            logger.info("Setting up transformer encoder decoder layers")
            self.setup_transformer_encoder_decoder_layers()
        
    
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
        
    def setup_transformer_encoder_decoder_layers(self):
        # Instead, create transformer layers:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.hidden_size,
            nhead=self.hparams.transformer_encoder_decoder_num_heads,
            dim_feedforward=self.hparams.transformer_encoder_decoder_hidden_size,
            dropout=self.hparams.dropout,
            batch_first=True,  
        )
        self.transformer_encoder = SharedTransformerEncoder(
            layer=encoder_layer,
            num_layers=self.transformer_encoder_decoder_num_layers,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hparams.hidden_size,
            nhead=self.hparams.transformer_encoder_decoder_num_heads,
            dim_feedforward=self.hparams.transformer_encoder_decoder_hidden_size,
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        self.transformer_decoder = SharedTransformerDecoder(
            layer=decoder_layer,
            num_layers=self.transformer_encoder_decoder_num_layers,
        )
        
        self.transformer_lstm_projection_layer = GatedTransformerLSTMProjectionUnit(
            hidden_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout
        )

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
        
        if self.hparams.use_transformer_variable_selection_networks:
            logger.info("Using TransformerVariableSelectionNetwork")
            VariableSelectionNetwork = TransformerVariableSelectionNetwork
        else:
            logger.info("Using PatchedVariableSelectionNetwork")
            VariableSelectionNetwork = PatchedVariableSelectionNetwork
        
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hparams.hidden_size,
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
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hparams.hidden_size),
                    self.hparams.hidden_size,
                    self.hparams.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hparams.hidden_size),
                        self.hparams.hidden_size,
                        self.hparams.dropout,
                    )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
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

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
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

        
        if self.hparams.use_transformer_encoder_decoder_layers:
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
            transformer_output_encoder = self.post_lstm_gate_encoder(transformer_encoder_output)
            transformer_output_encoder = self.post_lstm_add_norm_encoder(
                transformer_output_encoder, embeddings_varying_encoder
            )

            transformer_output_decoder = self.post_lstm_gate_decoder(transformer_decoder_output)
            transformer_output_decoder = self.post_lstm_add_norm_decoder(
                transformer_output_decoder, embeddings_varying_decoder
            )

            transformer_output = torch.cat([transformer_output_encoder, transformer_output_decoder], dim=1)

        # --------------------------------------------------------------------
        # Process with the LSTM layer as well
        # --------------------------------------------------------------------
        
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )

        # run local encoder
        lstm_encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder,
            (input_hidden, input_cell),
            lengths=encoder_lengths,
            enforce_sorted=False,
        )

        # run local decoder
        lstm_decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )
    
    
        lstm_output_encoder = self.post_lstm_gate_encoder(lstm_encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(
            lstm_output_encoder, embeddings_varying_encoder
        )

        lstm_output_decoder = self.post_lstm_gate_decoder(lstm_decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(
            lstm_output_decoder, embeddings_varying_decoder
        )

        # Combine last encoder and decoder outputs
        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        if self.hparams.use_transformer_encoder_decoder_layers:
            # Merge transformer and LSTM outputs
            fused_output = self.transformer_lstm_projection_layer(torch.cat([transformer_output, lstm_output], dim=-1))
        else:
            # Use the LSTM outputs as is
            fused_output = lstm_output
                
        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(fused_output,
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
            output, fused_output[:, max_encoder_length:]
        )
        
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
        self.training_batch_full_outputs = [] # only retain the data of the last batch
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        self.training_step_outputs.append(log) # from the base_model.py 
        self.training_batch_full_outputs.append((log, out))
        
        # Access and log the current learning rate from the first optimizer's first param group
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        
        return log

    def validation_step(self, batch, batch_idx):
        """
        Validation step: processes a validation batch, updates logging, and stores outputs.
        """
        self.validation_batch_full_outputs = [] # only retain the data of the last batch
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
        self.test_batch_full_outputs = [] # only retain the data of the last batch
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        self.testing_step_outputs.append(log)
        self.test_batch_full_outputs.append((log, out))
        return log

    # def plot_prediction(
    #     self,
    #     x: Dict[str, torch.Tensor],
    #     out: Dict[str, torch.Tensor],
    #     idx: int = 0,
    #     add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
    #     show_future_observed: bool = True,
    #     ax=None,
    #     quantiles_kwargs: Optional[Dict[str, Any]] = None,
    #     prediction_kwargs: Optional[Dict[str, Any]] = None,
    # ):
    #     """
    #     Plot prediction of prediction vs actuals

    #     Args:
    #         x: network input
    #         out: network output
    #         idx: index of prediction to plot
    #         add_loss_to_title: if to add loss to title or loss function to calculate. Can be either metrics,
    #             bool indicating if to use loss metric or tensor which contains losses for all samples.
    #             Calcualted losses are determined without weights. Default to False.
    #         show_future_observed: if to show actuals for future. Defaults to True.
    #         ax: matplotlib axes to plot on
    #         quantiles_kwargs (Dict[str, Any]): parameters for ``to_quantiles()`` of the loss metric.
    #         prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.

    #     Returns:
    #         matplotlib figure
    #     """  # noqa: E501
    #     if quantiles_kwargs is None:
    #         quantiles_kwargs = {}
    #     if prediction_kwargs is None:
    #         prediction_kwargs = {}

    #     from matplotlib import pyplot as plt

    #     # all true values for y of the first sample in batch
    #     encoder_targets = to_list(x["encoder_target"])
    #     decoder_targets = to_list(x["decoder_target"])

    #     y_raws = to_list(
    #         out["prediction"]
    #     )  # raw predictions - used for calculating loss
    #     y_hats = to_list(self.to_prediction(out, **prediction_kwargs))
    #     y_quantiles = to_list(self.to_quantiles(out, **quantiles_kwargs))

    #     # for each target, plot
    #     figs = []
    #     for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
    #         y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
    #     ):
    #         y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
    #         max_encoder_length = x["encoder_lengths"].max()
    #         y = torch.cat(
    #             (
    #                 y_all[: x["encoder_lengths"][idx]],
    #                 y_all[
    #                     max_encoder_length : (
    #                         max_encoder_length + x["decoder_lengths"][idx]
    #                     )
    #                 ],
    #             ),
    #         )
    #         # move predictions to cpu
    #         y_hat = y_hat.detach().cpu().to(torch.float32)[idx, : x["decoder_lengths"][idx]]
    #         y_quantile = y_quantile.detach().cpu().to(torch.float32)[idx, : x["decoder_lengths"][idx]]
    #         y_raw = y_raw.detach().cpu().to(torch.float32)[idx, : x["decoder_lengths"][idx]]

    #         # move to cpu
    #         y = y.detach().cpu().to(torch.float32)
    #         # create figure
    #         if ax is None:
    #             fig, ax = plt.subplots()
    #         else:
    #             fig = ax.get_figure()
    #         n_pred = y_hat.shape[0]
    #         x_obs = np.arange(-(y.shape[0] - n_pred), 0)
    #         x_pred = np.arange(n_pred)
    #         prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
    #         obs_color = next(prop_cycle)["color"]
    #         pred_color = next(prop_cycle)["color"]
    #         # plot observed history
    #         if len(x_obs) > 0:
    #             if len(x_obs) > 1:
    #                 plotter = ax.plot
    #             else:
    #                 plotter = ax.scatter
    #             plotter(x_obs, y[:-n_pred], label="observed", c=obs_color)
    #         if len(x_pred) > 1:
    #             plotter = ax.plot
    #         else:
    #             plotter = ax.scatter

    #         # plot observed prediction
    #         if show_future_observed:
    #             plotter(x_pred, y[-n_pred:], label=None, c=obs_color)

    #         # plot prediction
    #         plotter(x_pred, y_hat, label="predicted", c=pred_color)

    #         # plot predicted quantiles
    #         plotter(
    #             x_pred,
    #             y_quantile[:, y_quantile.shape[1] // 2],
    #             c=pred_color,
    #             alpha=0.15,
    #         )
    #         for i in range(y_quantile.shape[1] // 2):
    #             if len(x_pred) > 1:
    #                 ax.fill_between(
    #                     x_pred,
    #                     y_quantile[:, i],
    #                     y_quantile[:, -i - 1],
    #                     alpha=0.15,
    #                     fc=pred_color,
    #                 )
    #             else:
    #                 quantiles = torch.tensor(
    #                     [[y_quantile[0, i]], [y_quantile[0, -i - 1]]]
    #                 )
    #                 ax.errorbar(
    #                     x_pred,
    #                     y[[-n_pred]],
    #                     yerr=quantiles - y[-n_pred],
    #                     c=pred_color,
    #                     capsize=1.0,
    #                 )
    
    #         if add_loss_to_title is not False:
    #             if isinstance(add_loss_to_title, bool):
    #                 loss = self.loss
    #             elif isinstance(add_loss_to_title, torch.Tensor):
    #                 loss = add_loss_to_title.detach()[idx].item()
    #             elif isinstance(add_loss_to_title, Metric):
    #                 loss = add_loss_to_title
    #             else:
    #                 raise ValueError(
    #                     f"add_loss_to_title '{add_loss_to_title}'' is unkown"
    #                 )
    #             if isinstance(loss, MASE):
    #                 loss_value = loss(
    #                     y_raw[None], (y[-n_pred:][None], None), y[:n_pred][None]
    #                 )
    #             elif isinstance(loss, Metric):
    #                 try:
    #                     loss_value = loss(y_raw[None], (y[-n_pred:][None], None))
    #                 except Exception:
    #                     loss_value = "-"
    #             else:
    #                 loss_value = loss
    #             ax.set_title(f"Loss {loss_value}")
    #         ax.set_xlabel("Time index")
    #         fig.legend()
    #         figs.append(fig)

    #     # return multiple of target is a list, otherwise return single figure
    #     if isinstance(x["encoder_target"], (tuple, list)):
    #         return figs
    #     else:
    #         return fig