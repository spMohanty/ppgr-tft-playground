import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from copy import deepcopy

from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import GatedResidualNetwork, ResampleNorm

class SharedTransformerEncoder(nn.Module):
    def __init__(self, layer: nn.TransformerEncoderLayer, num_layers: int):
        """
        A transformer encoder that applies the same layer (weight sharing) repeatedly.
        Args:
            layer (nn.TransformerEncoderLayer): The transformer encoder layer to be shared.
            num_layers (int): How many times to apply the layer.
        """
        super().__init__()
        self.shared_layer = layer
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for _ in range(self.num_layers):
            output = self.shared_layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        return output

class SharedTransformerDecoder(nn.Module):
    def __init__(self, layer: nn.TransformerDecoderLayer, num_layers: int):
        """
        A transformer decoder that applies the same layer (weight sharing) repeatedly.
        Args:
            layer (nn.TransformerDecoderLayer): The transformer decoder layer to be shared.
            num_layers (int): How many times to apply the layer.
        """
        super().__init__()
        self.shared_layer = layer
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for _ in range(self.num_layers):
            output = self.shared_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return output

class GatedTransformerLSTMProjectionUnit(nn.Module):
    """Gated Linear Projection Unit to fuse transformer and LSTM outputs"""

    def __init__(self, hidden_size: int = None, dropout: float = None):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size * 2, hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x

class PatchedVariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        input_embedding_flags: Dict[str, bool] = None,
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, GatedResidualNetwork] = None,
        prescalers: Dict[str, nn.Linear] = None,
    ):
        """
        Calculate weights for ``num_inputs`` variables  which are each of size
        ``input_size``
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self._input_embedding_flags = (
            {} if input_embedding_flags is None else deepcopy(input_embedding_flags)
        )
        self.dropout = dropout
        self.context_size = context_size

        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    self.context_size,
                    residual=False,
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    residual=False,
                )
        if single_variable_grns is None:
            single_variable_grns = {}
        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        print(f"self.input_sizes : {self.input_sizes}")
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            elif self._input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = ResampleNorm(
                    input_size, self.hidden_size
                )
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    output_size=self.hidden_size,
                    dropout=self.dropout,
                )
            if prescalers is None:
                prescalers = {}
            if name in prescalers:  # reals need to be first scaled up
                self.prescalers[name] = prescalers[name]
            elif not self._input_embedding_flags.get(name, False):
                self.prescalers[name] = nn.Linear(1, input_size)

        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(
            size if name in self._input_embedding_flags else size
            for name, size in self.input_sizes.items()
        )

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: Dict[str, torch.Tensor], context: torch.Tensor = None):
        if self.num_inputs > 1:
            # transform single variables
            var_outputs = []
            weight_inputs = []
            for name in self.input_sizes.keys():
                # select embedding belonging to a single input
                variable_embedding = x[name]
                if name in self.prescalers:
                    variable_embedding = self.prescalers[name](variable_embedding)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)

            # calculate variable weights
            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            outputs = var_outputs * sparse_weights
            outputs = outputs.sum(dim=-1)
        elif self.num_inputs == 1:
            # for one input, do not perform variable selection but just encoding
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](
                variable_embedding
            )  # fast forward if only one variable
            if outputs.ndim == 3:  # -> batch size, time, hidden size, n_variables
                sparse_weights = torch.ones(
                    outputs.size(0), outputs.size(1), 1, 1, device=outputs.device
                )  #
            else:  # ndim == 2 -> batch size, hidden size, n_variables
                sparse_weights = torch.ones(
                    outputs.size(0), 1, 1, device=outputs.device
                )
        else:  # for no input
            outputs = torch.zeros(context.size(), device=context.device)
            if outputs.ndim == 3:  # -> batch size, time, hidden size, n_variables
                sparse_weights = torch.zeros(
                    outputs.size(0), outputs.size(1), 1, 0, device=outputs.device
                )
            else:  # ndim == 2 -> batch size, hidden size, n_variables
                sparse_weights = torch.zeros(
                    outputs.size(0), 1, 0, device=outputs.device
                )
        return outputs, sparse_weights


class TransformerVariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        input_embedding_flags: Dict[str, bool] = None,
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, nn.Module] = None,
        prescalers: Dict[str, nn.Module] = None,
    ):
        """
        Transformer-based variable selection network.
        
        Args:
            input_sizes: Dictionary mapping variable names to their input dimensions.
            hidden_size: The hidden dimension (this is used as the transformer model dimension).
            input_embedding_flags: (Ignored in this implementation.)
            dropout: Dropout probability.
            context_size: If provided, context will be projected and added to a learnable CLS token.
            single_variable_grns: (Ignored in this implementation.)
            prescalers: Optional dict mapping variable names to a prescaler (e.g. an nn.Linear)
                        that should be applied before the main variable embedding.
        """
        super().__init__()
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.context_size = context_size
        self.input_embedding_flags = input_embedding_flags if input_embedding_flags is not None else {}
        self.n_vars = len(input_sizes)
        self.single_variable = (self.n_vars == 1)

        # If provided, use the given prescalers; otherwise, initialize an empty ModuleDict.
        if prescalers is not None:
            self.prescalers = nn.ModuleDict(prescalers)
        else:
            self.prescalers = nn.ModuleDict()

        # Build a simple per-variable embedding layer mapping from the variable’s input dim to hidden_size.
        self.variable_embeddings = nn.ModuleDict()
        for name, size in input_sizes.items():
            self.variable_embeddings[name] = nn.Linear(size, hidden_size)

        # If context is provided, project it to hidden_size.
        if context_size is not None:
            self.context_proj = nn.Linear(context_size, hidden_size)
        else:
            self.context_proj = None

        # We'll use a fixed number of attention heads.
        self.n_head = 4  # You can adjust this as needed.

        # Learned CLS token used as the query for variable selection.
        # Shape: [1, 1, hidden_size] — later expanded to match the batch (and time) dimensions.
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Multi-head attention layer.
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=self.n_head, dropout=dropout, batch_first=True
        )

        # An optional feed-forward network and layer norm following the attention layer.
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, x: Dict[str, torch.Tensor], context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Dictionary of variable names to tensors.
               Each tensor can be of shape [batch, input_dim] or [batch, time, input_dim].
            context: Optional context tensor of shape [batch, context_size] or [batch, time, context_size].

        Returns:
            output: The combined variable embedding, of shape [batch, hidden_size] (if no time dimension)
                    or [batch, time, hidden_size] (if inputs are time-varying).
            sparse_weights: The variable selection weights (attention weights),
                            of shape [batch, n_vars] or [batch, time, n_vars].
        """
        # Determine whether inputs are time-varying based on one sample.
        sample_tensor = next(iter(x.values()))
        has_time = (sample_tensor.dim() == 3)

        # --- Case 1: Single variable input ---
        if self.single_variable:
            var_name = list(self.input_sizes.keys())[0]
            var_tensor = x[var_name]
            if var_tensor.dim() == 2:
                var_tensor = var_tensor.unsqueeze(1)  # Shape: [B, 1, input_dim]
            if var_name in self.prescalers:
                var_tensor = self.prescalers[var_name](var_tensor)
            # Embed the variable.
            output = self.variable_embeddings[var_name](var_tensor)  # [B, T, hidden_size]
            B, T, _ = output.shape
            # Create trivial sparse weights (all ones).
            sparse_weights = torch.ones(B, T, 1, device=output.device)
            # If time dimension is 1, squeeze it.
            if T == 1:
                output = output.squeeze(1)
                sparse_weights = sparse_weights.squeeze(1)
            return output, sparse_weights

        # --- Case 2: Multiple variable inputs ---
        # Process each variable: apply prescaler (if provided) then the embedding.
        embedded_vars = []
        for name, size in self.input_sizes.items():
            var_tensor = x[name]
            if var_tensor.dim() == 2:
                var_tensor = var_tensor.unsqueeze(1)  # [B, 1, input_dim]
            if name in self.prescalers:
                var_tensor = self.prescalers[name](var_tensor)
            embedded = self.variable_embeddings[name](var_tensor)  # [B, T, hidden_size]
            embedded_vars.append(embedded)

        # Stack along a new variable dimension so that tokens shape becomes [B, T, n_vars, hidden_size].
        tokens = torch.stack(embedded_vars, dim=2)
        B, T, n_vars, H = tokens.shape
        # Merge batch and time dimensions for transformer processing: [B*T, n_vars, H].
        tokens_reshaped = tokens.view(B * T, n_vars, H)

        # Create a CLS token for each instance.
        cls_tokens = self.cls_token.expand(B * T, -1, -1)  # [B*T, 1, H]
        # If a context tensor is provided, project and add it to the CLS token.
        if context is not None:
            if context.dim() == 3:
                # [B, T, context_size] -> [B*T, 1, context_size]
                context = context.reshape(B * T, 1, -1)
            elif context.dim() == 2:
                # [B, context_size] -> [B, 1, context_size] then repeat for each time step.
                context = context.unsqueeze(1).expand(B, T, -1).reshape(B * T, 1, -1)
            if self.context_proj is not None:
                context = self.context_proj(context)
            cls_tokens = cls_tokens + context

        # Concatenate the CLS token with the variable tokens.
        # attn_input: [B*T, 1+n_vars, H]
        attn_input = torch.cat([cls_tokens, tokens_reshaped], dim=1)

        # Use the CLS token as the query, and the full sequence as key/value.
        query = attn_input[:, :1, :]  # [B*T, 1, H]
        key = attn_input             # [B*T, 1+n_vars, H]
        value = attn_input

        # Apply multi-head attention.
        attn_output, attn_weights = self.mha(query, key, value)
        # attn_weights shape: [B*T, 1, 1+n_vars]. Discard the first column (self-attention of CLS).
        variable_selection_weights = attn_weights[:, :, 1:]  # [B*T, 1, n_vars]

        # Optionally, pass the output through a feed-forward network and add a residual connection.
        ff = self.feed_forward(attn_output)
        output = self.layer_norm(attn_output + ff)  # [B*T, 1, H]

        # Reshape back to [B, T, H].
        output = output.view(B, T, H)
        variable_selection_weights = variable_selection_weights.view(B, T, n_vars)

        # If time dimension is 1, squeeze it.
        if T == 1:
            output = output.squeeze(1)
            variable_selection_weights = variable_selection_weights.squeeze(1)

        return output, variable_selection_weights
