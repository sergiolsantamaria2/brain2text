import torch 
from torch import nn

class GRUDecoder(nn.Module):
    '''
    Defines the GRU decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                neural_dim,
                n_units,
                n_days,
                n_classes,
                rnn_dropout=0.0,
                input_dropout=0.0,
                n_layers=5,
                patch_size=0,
                patch_stride=0,
                # New: post-RNN head (training improvement)
                head_type: str = "none",          # "none" | "resffn"
                head_num_blocks: int = 0,         # e.g., 1 or 2
                head_norm: str = "none",          # "bn" | "layernorm" | "rmsnorm" | "none"
                head_dropout: float = 0.0,
                head_activation: str = "gelu",
                # New: speckled masking (coordinated dropout)
                input_speckle_p: float = 0.0,
                input_speckle_mode: str = "feature",
                ):

        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes 
        rnn_dropout    (float) - percentage of units to droupout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers 
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step 
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input 
        '''
        super(GRUDecoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.head_type = str(head_type)
        self.head_num_blocks = int(head_num_blocks)
        self.head_norm = str(head_norm)
        self.head_dropout = float(head_dropout)
        self.head_activation = str(head_activation)

        self.input_speckle_p = float(input_speckle_p)
        self.input_speckle_mode = str(input_speckle_mode)


        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

       # Day-specific affine parameters (vectorized, compile-friendly)
        self.day_weights = nn.Parameter(
            torch.eye(self.neural_dim).unsqueeze(0).repeat(self.n_days, 1, 1)
        )  # (n_days, D, D)

        self.day_biases = nn.Parameter(
            torch.zeros(self.n_days, self.neural_dim)
        )  # (n_days, D)


        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout, 
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = False,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Optional post-GRU head
        ht = self.head_type.lower()
        if ht == "none" or self.head_num_blocks <= 0:
            self.head = nn.Identity()
        elif ht in ("resffn", "ffn"):
            self.head = nn.Sequential(*[
                ResidualFFNBlock(
                    d=self.n_units,
                    norm_type=self.head_norm,
                    dropout=self.head_dropout,
                    activation=self.head_activation,
                )
                for _ in range(self.head_num_blocks)
            ])
        else:
            raise ValueError(f"Unknown head_type={self.head_type}. Use: none, resffn.")

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_ids = day_idx.view(-1).long()  # (B,)

        day_weights = self.day_weights.index_select(0, day_ids)          # (B, D, D)
        day_biases  = self.day_biases.index_select(0, day_ids).unsqueeze(1)  # (B, 1, D)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)


        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0: 
  
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Speckled masking (training only)
        if self.training and self.input_speckle_p > 0:
            x = speckle_mask(x, self.input_speckle_p, self.input_speckle_mode)

        
        # Pass input through RNN 
        output, hidden_states = self.gru(x, states)

        # Optional post-GRU head
        output = self.head(output)

        # Compute logits
        logits = self.out(output)

        
        if return_state:
            return logits, hidden_states
        
        return logits
        

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


class MyBatchNorm1d(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(d, eps=eps, momentum=momentum)

    def forward(self, x):
        if x.dim() == 3:  # (B,T,D)
            b, t, d = x.shape
            y = x.reshape(b * t, d)
            y = self.bn(y)
            return y.reshape(b, t, d)
        elif x.dim() == 2:  # (B,D)
            return self.bn(x)
        else:
            raise ValueError(f"MyBatchNorm1d expected 2D or 3D input, got shape={tuple(x.shape)}")

def build_time_norm(norm_type: str, d: int) -> nn.Module:
    norm_type = (norm_type or "none").lower()
    if norm_type == "bn":
        return MyBatchNorm1d(d)
    if norm_type == "layernorm":
        return nn.LayerNorm(d)
    if norm_type == "rmsnorm":
        return RMSNorm(d)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type={norm_type}. Use one of: bn, layernorm, rmsnorm, none.")

def get_activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation={name}. Use one of: gelu, relu, silu.")


def speckle_mask(x: torch.Tensor, p: float, mode: str) -> torch.Tensor:
    """
    Coordinated dropout / speckled masking.
    x: (B,T,D)
    mode:
      - 'feature': drop entire features across all timesteps (mask shape Bx1xD)
      - 'time':    drop entire timesteps across all features (mask shape BxTx1)
      - 'both':    elementwise (BxTxD)  (usually less stable; keep for ablation)
    """
    if p <= 0.0:
        return x
    mode = (mode or "feature").lower()
    B, T, D = x.shape
    if mode == "feature":
        mask = torch.rand(B, 1, D, device=x.device) < p
    elif mode == "time":
        mask = torch.rand(B, T, 1, device=x.device) < p
    elif mode == "both":
        mask = torch.rand(B, T, D, device=x.device) < p
    else:
        raise ValueError(f"Unknown speckle mode={mode}. Use: feature, time, both.")
    return x.masked_fill(mask, 0.0)


class ResidualFFNBlock(nn.Module):
    """
    Simple GPT-style MLP block without attention:
      x <- x + Dropout(Act(Linear(Norm(x))))
    Works on (B,T,D).
    """
    def __init__(self, d: int, norm_type: str, dropout: float, activation: str):
        super().__init__()
        self.norm = build_time_norm(norm_type, d)
        self.lin = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.lin.weight)
        self.act = get_activation(activation)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(self.norm(x))
        y = self.act(y)
        y = self.drop(y)
        return x + y


class ResLSTMSublayer(nn.Module):
    """
    One residual BiLSTM sublayer:
      - pre_norm: y = LSTM(norm(x)); x = x + dropout(y)
      - post_norm: y = LSTM(x); x = norm(x + dropout(y))
    """
    def __init__(
        self,
        d: int,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        norm_type: str = "bn",
        pre_norm: bool = False,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        assert d % 2 == 0, f"ResLSTM requires even d, got d={d}"
        self.pre_norm = bool(pre_norm)

        self.norm = build_time_norm(norm_type, d)
        self.residual_dropout = nn.Dropout(p=float(residual_dropout))

        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=d // 2,
            num_layers=int(lstm_layers),
            dropout=float(lstm_dropout) if int(lstm_layers) > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x_in = self.norm(x)
            y, _ = self.lstm(x_in)
            y = self.residual_dropout(y)
            return x + y
        else:
            y, _ = self.lstm(x)
            y = self.residual_dropout(y)
            return self.norm(x + y)


class ResLSTMBlock(nn.Module):
    """
    Notebook-style block = 2 residual BiLSTM sublayers.
    """
    def __init__(
        self,
        d: int,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        norm_type: str = "bn",
        pre_norm: bool = False,
        residual_dropout: float = 0.0,
        sublayers_per_block: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ResLSTMSublayer(
                d=d,
                lstm_layers=lstm_layers,
                lstm_dropout=lstm_dropout,
                norm_type=norm_type,
                pre_norm=pre_norm,
                residual_dropout=residual_dropout,
            )
            for _ in range(int(sublayers_per_block))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResLSTMDecoder(nn.Module):
    """
    Decoder:
      - day-specific affine
      - patching
      - projection to n_units
      - stack of ResLSTMBlocks
      - output head
    """
    def __init__(
        self,
        neural_dim: int,
        n_units: int,
        n_days: int,
        n_classes: int,
        rnn_dropout: float,
        input_dropout: float,
        n_layers: int,          # kept for compatibility
        patch_size: int,
        patch_stride: int,
        # knobs
        reslstm_num_blocks: int = 1,
        reslstm_sublayers_per_block: int = 2,
        reslstm_lstm_layers: int = 2,
        reslstm_lstm_dropout: float = 0.1,
        reslstm_norm: str = "bn",
        reslstm_pre_norm: bool = False,
        reslstm_residual_dropout: float = 0.0,
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

        # Day-specific affine
        self.day_layer_activation = nn.Softsign()
        self.day_layer_dropout = nn.Dropout(p=float(input_dropout))

        # en __init__
        self.day_weights = nn.Parameter(torch.eye(self.neural_dim).unsqueeze(0).repeat(self.n_days, 1, 1))
        self.day_biases  = nn.Parameter(torch.zeros(self.n_days, self.neural_dim))


        # Patching => flatten => proj
        in_dim = neural_dim * self.patch_size if self.patch_size > 0 else neural_dim
        self.in_proj = nn.Linear(in_dim, n_units)
        nn.init.xavier_uniform_(self.in_proj.weight)

        # Stack blocks
        self.reslstm = nn.Sequential(*[
            ResLSTMBlock(
                d=n_units,
                lstm_layers=int(reslstm_lstm_layers),
                lstm_dropout=float(reslstm_lstm_dropout),
                norm_type=str(reslstm_norm),
                pre_norm=bool(reslstm_pre_norm),
                residual_dropout=float(reslstm_residual_dropout),
                sublayers_per_block=int(reslstm_sublayers_per_block),
            )
            for _ in range(int(reslstm_num_blocks))
        ])

        self.dropout = nn.Dropout(p=float(rnn_dropout))
        self.out = nn.Linear(n_units, n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, features: torch.Tensor, day_indicies: torch.Tensor) -> torch.Tensor:
        # Usar las variables que realmente entran a la función
        x = features

        # Vectorized day indexing (no .tolist() sync)
        day_ids = day_indicies.view(-1).long()                 # (B,)
        W = self.day_weights.index_select(0, day_ids)          # (B,D,D)
        b = self.day_biases.index_select(0, day_ids).unsqueeze(1)  # (B,1,D)

        # Day-specific affine
        x = torch.einsum("btd,bdk->btk", x, W) + b
        x = self.day_layer_activation(x)
        x = self.day_layer_dropout(x)  # dropout(p=0) => identity

        # Patching
        if self.patch_size > 0:
            ps = self.patch_size
            st = self.patch_stride
            x = x.unfold(dimension=1, size=ps, step=st)        # (B, T', C, ps)
            x = x.permute(0, 1, 3, 2).contiguous()             # (B, T', ps, C)
            x = x.view(x.size(0), x.size(1), -1)               # (B, T', ps*C)

        # Project + ResLSTM blocks + head
        x = self.in_proj(x)                                    # (B, T', n_units)
        x = self.reslstm(x)                                    # (B, T', n_units)
        x = self.dropout(x)
        logits = self.out(x)                                   # (B, T', n_classes)
        return logits



from brain2text.xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from brain2text.xlstm.blocks.slstm.block import sLSTMBlockConfig
from brain2text.xlstm.blocks.slstm.layer import sLSTMLayerConfig


import torch
from torch import nn

class XLSTMDecoder(nn.Module):
    def __init__(
        self,
        neural_dim: int,
        n_units: int,
        n_days: int,
        n_classes: int,
        rnn_dropout: float,
        input_dropout: float,
        n_layers: int,
        patch_size: int,
        patch_stride: int,
       xlstm_num_blocks: int = None,
        xlstm_num_heads: int = 4,
        xlstm_conv1d_kernel_size: int = 4,
        xlstm_dropout: float = None,
        xlstm_backend: str = "cuda",  # "cuda" or "vanilla"
        # --- NEW: post-backbone head (match GRU feature set) ---
        head_type: str = "none",          # "none" | "resffn"
        head_num_blocks: int = 0,
        head_norm: str = "none",          # "bn" | "layernorm" | "rmsnorm" | "none"
        head_dropout: float = 0.0,
        head_activation: str = "gelu",
        # --- NEW: speckled masking (match GRU feature set) ---
        input_speckle_p: float = 0.0,
        input_speckle_mode: str = "feature",
    ):
        super().__init__()

        self.neural_dim = int(neural_dim)
        self.n_units = int(n_units)
        self.n_days = int(n_days)
        self.n_classes = int(n_classes)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

        if xlstm_num_blocks is None:
            xlstm_num_blocks = int(n_layers)
        if xlstm_dropout is None:
            xlstm_dropout = float(rnn_dropout)

        self.xlstm_num_blocks = int(xlstm_num_blocks)
        self.xlstm_num_heads = int(xlstm_num_heads)
        self.xlstm_conv1d_kernel_size = int(xlstm_conv1d_kernel_size)
        self.xlstm_dropout = float(xlstm_dropout)
        self.xlstm_backend = str(xlstm_backend)

        # Head + speckle knobs
        self.head_type = str(head_type)
        self.head_num_blocks = int(head_num_blocks)
        self.head_norm = str(head_norm)
        self.head_dropout = float(head_dropout)
        self.head_activation = str(head_activation)

        self.input_speckle_p = float(input_speckle_p)
        self.input_speckle_mode = str(input_speckle_mode)

        if self.n_units % self.xlstm_num_heads != 0:
            raise ValueError(
                f"xlstm_num_heads must divide n_units. Got n_units={self.n_units}, heads={self.xlstm_num_heads}"
            )

        # Day-specific affine (same pattern as GRU)
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.Parameter(torch.eye(self.neural_dim).unsqueeze(0).repeat(self.n_days, 1, 1))
        self.day_biases = nn.Parameter(torch.zeros(self.n_days, self.neural_dim))
        self.day_layer_dropout = nn.Dropout(p=float(input_dropout))

        # Patching projection: (B,T,C) -> (B,T', patch_size*C) then -> n_units
        in_dim = (self.patch_size * self.neural_dim) if self.patch_size > 0 else self.neural_dim
        self.in_proj = nn.Linear(in_dim, self.n_units, bias=True)
        nn.init.xavier_uniform_(self.in_proj.weight)

        # sLSTM-only stack
        slstm_layer_cfg = sLSTMLayerConfig(
            embedding_dim=self.n_units,
            num_heads=self.xlstm_num_heads,
            conv1d_kernel_size=self.xlstm_conv1d_kernel_size,
            dropout=self.xlstm_dropout,
        )
        slstm_block_cfg = sLSTMBlockConfig(slstm=slstm_layer_cfg, feedforward=None)

        stack_cfg = xLSTMBlockStackConfig(
            mlstm_block=None,
            slstm_block=slstm_block_cfg,
            context_length=-1,
            num_blocks=self.xlstm_num_blocks,
            embedding_dim=self.n_units,
            add_post_blocks_norm=True,
            bias=False,
            dropout=self.xlstm_dropout,
            slstm_at="all",
        )
        self.xlstm = xLSTMBlockStack(stack_cfg)

        # Optional post-xLSTM head (same design as your GRU head)
        ht = self.head_type.lower()
        if ht == "none" or self.head_num_blocks <= 0:
            self.head = nn.Identity()
        elif ht in ("resffn", "ffn"):
            self.head = nn.Sequential(*[
                ResidualFFNBlock(
                    d=self.n_units,
                    norm_type=self.head_norm,
                    dropout=self.head_dropout,
                    activation=self.head_activation,
                )
                for _ in range(self.head_num_blocks)
            ])
        else:
            raise ValueError(f"Unknown head_type={self.head_type}. Use: none, resffn.")

        self.dropout = nn.Dropout(p=float(rnn_dropout))
        self.out = nn.Linear(self.n_units, self.n_classes, bias=True)
        nn.init.xavier_uniform_(self.out.weight)

    def _apply_day_layer(self, x: torch.Tensor, day_indicies: torch.Tensor) -> torch.Tensor:
        day_ids = day_indicies.view(-1).long()  # (B,)
        W = self.day_weights.index_select(0, day_ids)              # (B, D, D)
        b = self.day_biases.index_select(0, day_ids).unsqueeze(1)  # (B, 1, D)

        x = torch.einsum("btd,bdk->btk", x, W) + b
        x = self.day_layer_activation(x)
        x = self.day_layer_dropout(x)  # dropout(p=0) is identity
        return x

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size <= 0:
            return x
        B, T, C = x.shape
        ps = self.patch_size
        st = self.patch_stride
        if st <= 0:
            raise ValueError(f"Invalid patch_stride={st} with patch_size={ps}")

        x = x.unfold(dimension=1, size=ps, step=st)     # (B, T', C, ps)
        x = x.permute(0, 1, 3, 2).contiguous()          # (B, T', ps, C)
        x = x.view(B, x.shape[1], ps * C)               # (B, T', ps*C)
        return x

    def forward(self, features: torch.Tensor, day_indicies: torch.Tensor) -> torch.Tensor:
        # features: (B,T,C)
        x = self._apply_day_layer(features, day_indicies)
        x = self._patchify(x)

        # Speckled masking (training only) - applied on the (possibly patched) input sequence
        if self.training and self.input_speckle_p > 0:
            x = speckle_mask(x, self.input_speckle_p, self.input_speckle_mode)

        x = self.in_proj(x)      # (B, T', n_units)
        x = self.xlstm(x)        # (B, T', n_units)
        x = self.head(x)         # optional residual FFN blocks
        x = self.dropout(x)
        logits = self.out(x)     # (B, T', n_classes)
        return logits
