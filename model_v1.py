import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import math

# We import the original base class:
from pytorch_forecasting import (
    TemporalFusionTransformer,
)

from pytorch_forecasting.utils import create_mask
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,  # we won't use this for variable selection, but keep for static gating if needed
    InterpretableMultiHeadAttention,
)

###################################################################
# 1) Per-feature Attention Module
###################################################################
class FeatureAttentionPerFeature(nn.Module):
    """
    Attention-based variable selection that handles multiple features,
    each possibly having a *different* embedding dimension.

    At each call, we receive a list of feature tensors, each shape:
      (batch_size, feature_dim_i)
    We learn a separate key-projection and value-projection for each feature.
    We also have one shared query vector (of size 'hidden_size').

    We compute:
       K_i = linear_key_i( features[i] )   -> shape (batch, hidden_size)
       V_i = linear_value_i( features[i] ) -> shape (batch, hidden_size)

    Then attention score for feature i is:
       score_i = dot(query, K_i) / sqrt(hidden_size)

    We take softmax across i (all features) to get attention weights alpha_i.
    Output is the weighted sum of V_i (batch, hidden_size), plus interpretability weights.
    """

    def __init__(self, feature_dims: List[int], hidden_size: int, dropout: float = 0.1):
        """
        Args:
            feature_dims: list of input dims, one for each feature
            hidden_size: dimension to which each feature is projected (and final output size)
            dropout: dropout probability
        """
        super().__init__()
        self.num_features = len(feature_dims)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        # create key/value linear layers per feature
        self.key_projections = nn.ModuleList([
            nn.Linear(dim, hidden_size, bias=False) for dim in feature_dims
        ])
        self.value_projections = nn.ModuleList([
            nn.Linear(dim, hidden_size, bias=False) for dim in feature_dims
        ])

        # single learnable query vector
        self.query = nn.Parameter(torch.randn(hidden_size))

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: list of length num_features,
                      each Tensor shape (batch_size, feature_dim_i)

        Returns:
            combined: (batch_size, hidden_size) - weighted sum of the values
            attn_weights: (batch_size, num_features) - attention distribution
        """
        batch_size = features[0].size(0)

        # 1) Project each feature to K, V
        K_list = []
        V_list = []
        for i in range(self.num_features):
            K_list.append(self.key_projections[i](features[i]))   # (batch, hidden_size)
            V_list.append(self.value_projections[i](features[i])) # (batch, hidden_size)

        # stack across features => shape (batch, num_features, hidden_size)
        K = torch.stack(K_list, dim=1)
        V = torch.stack(V_list, dim=1)

        # 2) Compute dot(query, K_i)
        # expand query => shape (1,1,hidden_size), then broadcast
        query = self.query.unsqueeze(0).unsqueeze(1)  # (1,1,hidden_size)
        attn_scores = (K * query).sum(dim=-1) / math.sqrt(self.hidden_size)  # (batch, num_features)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, num_features)

        # 3) Weighted sum of V => shape (batch, hidden_size)
        combined = torch.bmm(attn_weights.unsqueeze(1), V).squeeze(1)
        combined = self.dropout(combined)

        return combined, attn_weights


###################################################################
# 2) LocalSelfAttentionEncoderLayer remains unchanged
###################################################################
class LocalSelfAttentionEncoderLayer(nn.Module):
    """
    One transformer encoder layer that applies local (windowed) self-attention
    to capture short-term temporal patterns, replacing the local LSTM in original TFT.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, window: int = None):
        """
        Args:
            d_model: hidden dimension
            n_heads: number of attention heads
            dropout: dropout probability
            window: if not None, restrict self-attention to +/- window steps from each position
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch, seq_len, d_model)
        Returns: shape (batch, seq_len, d_model)
        """
        seq_len = x.size(1)

        # 1) Build local attention mask if window is set
        # shape for attn_mask in MultiheadAttention: (seq_len, seq_len)
        # We'll fill out-of-window positions with -inf to block them
        attn_mask = None
        if self.window is not None:
            attn_mask = x.new_ones(seq_len, seq_len) * float('-inf')
            for i in range(seq_len):
                low = max(0, i - self.window)
                high = min(seq_len, i + self.window + 1)
                attn_mask[i, low:high] = 0.0

        # 2) Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x2 = self.norm1(x + self.dropout(attn_out))

        # 3) Feed-forward
        ff_out = self.ff(x2)
        x3 = self.norm2(x2 + self.dropout(ff_out))

        return x3


###################################################################
# 3) PPGRTemporalFusionTransformer with per-feature attention
###################################################################
class PPGRTemporalFusionTransformer(TemporalFusionTransformer):
    """
    A purely transformer-native version of the Temporal Fusion Transformer (TFT).
    - Replaces LSTM with local transformer encoder/decoder
    - Replaces GRN-based variable selection with attention-based feature selection
      WITHOUT forcing all features to share the same embedding dim (Approach B).
    - Uses cross-attention to incorporate static features (static enrichment)
    - Preserves interpretability via attention weights (variable selection + cross-attention)
    - Optionally uses local attention windows to replicate LSTM's short-term inductive bias
    """

    def __init__(
        self,
        attention_head_size: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        dim_feedforward: int = 128,  # for the internal feed-forward in encoder/decoder
        local_attention_window: int = None,  # if not None, restricts attention to +/- window steps
        **kwargs,
    ):
        """
        Args:
            attention_head_size: number of heads for local/self attention
            n_encoder_layers: number of local self-attention layers for the past encoder
            n_decoder_layers: number of layers in the forecasting decoder
            dim_feedforward: dimension for feed-forward sublayers in the encoder/decoder
            local_attention_window: if set, use local windowed attention in the encoder
            **kwargs: same as in the base TemporalFusionTransformer
        """
        super().__init__(**kwargs)
        
        # Setup variables needed for the Metrics Callback
        self.validation_batch_full_outputs = []
        self.test_batch_full_outputs = []        

        # -------------------
        # Remove or ignore LSTM modules from the parent
        # -------------------
        del self.lstm_encoder
        del self.lstm_decoder

        # -------------
        # Setup per-feature attention modules
        # -------------
        #
        #  For static variables:
        #    - first gather their embedding dims
        #    - create a FeatureAttentionPerFeature with those dims
        #
        #  For encoder time-varying variables:
        #    - gather embedding dims
        #    - create FeatureAttentionPerFeature
        #
        #  For decoder time-varying variables:
        #    - gather embedding dims
        #
        # We'll store them in self.static_feat_attention, self.encoder_feat_attention, self.decoder_feat_attention
        #

        # Filter static variables that exist in input_embeddings.output_size
        self.filtered_static_variables = [feat for feat in self.static_variables if feat in self.input_embeddings.output_size]

        # Build a per-feature attention for static feats if any
        if len(self.filtered_static_variables) > 0:
            static_feature_dims = [self.input_embeddings.output_size[f] for f in self.filtered_static_variables]
            self.static_feat_attention = FeatureAttentionPerFeature(
                feature_dims=static_feature_dims,
                hidden_size=self.hparams.hidden_size,
                dropout=self.hparams.dropout,
            )
        else:
            self.static_feat_attention = None

        # Build per-feature attention for the *time-varying* encoder feats
        # (We do the same for the decoder feats.)
        self.encoder_feature_dims = [self.input_embeddings.output_size[f] for f in self.encoder_variables if f in self.input_embeddings.output_size]
        self.encoder_feat_attention = None
        if len(self.encoder_feature_dims) > 0:
            self.encoder_feat_attention = FeatureAttentionPerFeature(
                feature_dims=self.encoder_feature_dims,
                hidden_size=self.hparams.hidden_size,
                dropout=self.hparams.dropout,
            )

        self.decoder_feature_dims = [self.input_embeddings.output_size[f] for f in self.decoder_variables if f in self.input_embeddings.output_size]
        self.decoder_feat_attention = None
        if len(self.decoder_feature_dims) > 0:
            self.decoder_feat_attention = FeatureAttentionPerFeature(
                feature_dims=self.decoder_feature_dims,
                hidden_size=self.hparams.hidden_size,
                dropout=self.hparams.dropout,
            )

        # -------------------
        # LOCAL SELF-ATTENTION ENCODER
        # We'll stack n_encoder_layers of local self-attention
        # This captures short-term patterns in the historical segment
        # -------------------
        self.encoder_layers = nn.ModuleList([
            LocalSelfAttentionEncoderLayer(
                d_model=self.hparams.hidden_size,
                n_heads=attention_head_size,
                dropout=self.hparams.dropout,
                window=local_attention_window,
            )
            for _ in range(n_encoder_layers)
        ])

        # -------------------
        # CROSS-ATTENTION FOR STATIC ENRICHMENT
        # We'll do a small multihead attention from the encoder to static features
        # so each time step can incorporate static info
        # -------------------
        self.static_enrichment_attn = nn.MultiheadAttention(
            embed_dim=self.hparams.hidden_size,
            num_heads=1,  # single-head for simpler interpretability
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        self.static_enrichment_norm = nn.LayerNorm(self.hparams.hidden_size)

        # -------------------
        # DECODER: We'll do a standard transformer decoder approach:
        # 1) masked self-attn among forecast steps
        # 2) cross-attn to the encoder outputs
        # We'll build a small Transformer-like block repeated n_decoder_layers times.
        # -------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hparams.hidden_size,
            nhead=attention_head_size,
            dim_feedforward=dim_feedforward,
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_decoder_layers,
        )

    ########################################################################
    # Helper function: do time-distributed per-feature attention
    ########################################################################
    def _apply_time_distributed_attention(
        self,
        feat_attention_module: FeatureAttentionPerFeature,
        feats_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies per-feature attention across multiple time steps by looping.
        Each item in `feats_list` is shape (batch, time, dim_i) for feature i.
        We do a for-loop over time steps to collect (batch, dim_i) for each i,
        pass it to the attention module, then stack results.

        Args:
            feat_attention_module: a FeatureAttentionPerFeature instance
            feats_list: list of length n_feats,
                        each shape = (batch, time, d_i)

        Returns:
            combined_all_t: (batch, time, hidden_size)
            attn_weights_all_t: (batch, time, n_feats)
        """
        # feats_list[i] => (batch, time, d_i)
        batch_size = feats_list[0].size(0)
        seq_len = feats_list[0].size(1)
        n_feats = len(feats_list)
        hidden_size = feat_attention_module.hidden_size

        # We'll store the combined representation for each time step
        combined_outputs = []
        attn_weight_list = []

        for t in range(seq_len):
            # gather each feature i at time t => shape (batch, d_i)
            per_time_features = [feats_list[i][:, t, :] for i in range(n_feats)]

            # apply attention
            combined_t, attn_weights_t = feat_attention_module(per_time_features)
            # combined_t: (batch, hidden_size)
            # attn_weights_t: (batch, n_feats)

            combined_outputs.append(combined_t)
            attn_weight_list.append(attn_weights_t)

        # stack across time => shape (batch, time, hidden_size)
        combined_all_t = torch.stack(combined_outputs, dim=1)
        # same for attn => shape (batch, time, n_feats)
        attn_weights_all_t = torch.stack(attn_weight_list, dim=1)

        return combined_all_t, attn_weights_all_t

    ########################################################################
    # Main forward
    ########################################################################
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing:
          - per-feature attention-based variable selection (static + time-varying)
          - local self-attention for the encoder
          - cross-attention with static features
          - transformer decoder for future steps
          - optional final interpretative multi-head attention (from parent)
        """
        # 1) parse input
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        max_encoder_length = int(encoder_lengths.max())

        # Concatenate cat/cont across time
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        batch_size, total_time, _ = x_cont.shape

        # 2) build embeddings for each feature (the parent class does this)
        input_vectors = self.input_embeddings(x_cat)
        # incorporate continuous variables from x_cont
        for idx, name in enumerate(self.hparams.x_reals):
            cont_col = x_cont[..., idx].unsqueeze(-1)  # (batch, total_time, 1)
            if name in self.prescalers:
                cont_col = self.prescalers[name](cont_col)
            # store the scaled input
            input_vectors[name] = cont_col

        # Now input_vectors is a dict {feature_name: (batch, total_time, embedding_dim_for_that_feature)}

        #####################################################################
        # 3) STATIC variable selection
        #####################################################################
        if self.static_feat_attention is not None and len(self.filtered_static_variables) > 0:
            # gather each static feature as shape (batch, 1, dim_i) => then squeeze time dimension
            # we store them in a list
            static_per_feature = []
            for feat_name in self.filtered_static_variables:
                feat_val = input_vectors[feat_name][:, 0:1]  # (batch, 1, dim_i)
                feat_val_squeezed = feat_val.squeeze(1)     # (batch, dim_i)
                static_per_feature.append(feat_val_squeezed)

            # pass list to per-feature attention
            static_context, static_weights = self.static_feat_attention(static_per_feature)
            # static_context: (batch, hidden_size)
            # static_weights: (batch, n_static_vars)
        else:
            # no static variables => fill with zeros
            static_context = torch.zeros(
                (batch_size, self.hparams.hidden_size), device=x_cont.device, dtype=x_cont.dtype
            )
            static_weights = torch.zeros((batch_size, 0), device=x_cont.device, dtype=x_cont.dtype)

        #####################################################################
        # 4) ENCODER variable selection (historical time steps)
        #####################################################################
        # gather the time-varying features for the encoder portion
        encoder_feats_list = []
        for feat_name in self.encoder_variables:
            feat_val = input_vectors[feat_name][:, :max_encoder_length]  # (batch, max_encoder_length, dim)
            encoder_feats_list.append(feat_val)

        if self.encoder_feat_attention is not None and len(encoder_feats_list) > 0:
            # apply time-distributed attention
            encoder_selected, encoder_sparse_weights = self._apply_time_distributed_attention(
                self.encoder_feat_attention, encoder_feats_list
            )
            # encoder_selected: (batch, max_encoder_length, hidden_size)
            # encoder_sparse_weights: (batch, max_encoder_length, n_encoder_feats)
        else:
            # no encoder variables
            encoder_selected = torch.zeros(
                (batch_size, max_encoder_length, self.hparams.hidden_size),
                device=x_cont.device, dtype=x_cont.dtype,
            )
            encoder_sparse_weights = torch.zeros((batch_size, max_encoder_length, 0), device=x_cont.device)

        #####################################################################
        # 5) DECODER variable selection (future time steps)
        #####################################################################
        decoder_length = total_time - max_encoder_length
        decoder_feats_list = []
        for feat_name in self.decoder_variables:
            feat_val = input_vectors[feat_name][:, max_encoder_length:]  # (batch, dec_len, dim)
            decoder_feats_list.append(feat_val)

        if self.decoder_feat_attention is not None and len(decoder_feats_list) > 0:
            decoder_selected, decoder_sparse_weights = self._apply_time_distributed_attention(
                self.decoder_feat_attention, decoder_feats_list
            )
            # shape: (batch, dec_len, hidden_size), (batch, dec_len, n_decoder_feats)
        else:
            decoder_selected = torch.zeros(
                (batch_size, decoder_length, self.hparams.hidden_size),
                device=x_cont.device, dtype=x_cont.dtype
            )
            decoder_sparse_weights = torch.zeros((batch_size, decoder_length, 0), device=x_cont.device)

        #####################################################################
        # 6) LOCAL SELF-ATTENTION ENCODER
        #####################################################################
        enc_padding_mask = create_mask(max_encoder_length, encoder_lengths)  # shape (batch, max_encoder_length)
        x_enc = encoder_selected
        for layer in self.encoder_layers:
            x_enc = layer(x_enc)  # shape remains (batch, max_encoder_length, hidden_size)

        #####################################################################
        # 7) STATIC CROSS-ATTENTION (Static Enrichment)
        #####################################################################
        # treat static_context as single-vector memory
        static_context_3d = static_context.unsqueeze(1)  # (batch, 1, hidden_size)
        static_mem = static_context_3d.expand(-1, x_enc.size(1), -1)  # (batch, time, hidden_size)

        static_out, _ = self.static_enrichment_attn(x_enc, static_mem, static_mem)
        x_enc = self.static_enrichment_norm(x_enc + static_out)

        #####################################################################
        # 8) DECODER (transformer decoder) for the forecast steps
        #####################################################################
        x_dec = decoder_selected
        dec_len = x_dec.size(1)

        # causal mask for future steps
        dec_mask = torch.triu(
            torch.ones(dec_len, dec_len, device=x_dec.device, dtype=torch.bool),
            diagonal=1
        )

        # build padding masks
        memory_key_padding_mask = enc_padding_mask  # for encoder
        dec_padding_mask = create_mask(dec_len, decoder_lengths)  # for decoder

        x_dec_out = self.transformer_decoder(
            x_dec,
            x_enc,
            tgt_mask=dec_mask,
            tgt_key_padding_mask=dec_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        #####################################################################
        # 9) Final Interpretable Multi-Head Attention from parent
        #####################################################################
        full_seq = torch.cat([x_enc, x_dec_out], dim=1)  # (batch, max_enc_len + dec_len, hidden_size)
        query_part = full_seq[:, max_encoder_length:]    # (batch, dec_len, hidden_size)

        attn_out, attn_w = self.multihead_attn(
            q=query_part,
            k=full_seq,
            v=full_seq,
            mask=self.get_attention_mask(encoder_lengths, decoder_lengths),
        )
        attn_out = self.post_attn_gate_norm(attn_out, query_part)
        attn_out = self.pos_wise_ff(attn_out)
        attn_out = self.pre_output_gate_norm(attn_out, query_part)

        #####################################################################
        # 10) final linear to produce predictions
        #####################################################################
        if self.n_targets > 1:
            output = [layer(attn_out) for layer in self.output_layer]
        else:
            output = self.output_layer(attn_out)

        #####################################################################
        # 11) build the output dict, including interpretability
        #####################################################################
        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            encoder_attention=attn_w[..., :max_encoder_length],  # attend to encoder portion
            decoder_attention=attn_w[..., max_encoder_length:],  # attend to decoder portion
            static_variables=static_weights.unsqueeze(1),  # (batch, 1, n_static_vars)
            encoder_variables=encoder_sparse_weights.unsqueeze(-2),  # (batch, time, 1, n_enc_feats)
            decoder_variables=decoder_sparse_weights.unsqueeze(-2),  # (batch, time, 1, n_dec_feats)
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )

    ###################################################################
    # 12) validation_step/test_step remain as in your original snippet
    ###################################################################
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
