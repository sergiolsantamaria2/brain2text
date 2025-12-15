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
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 0,
                 patch_stride = 0,
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

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

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
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

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

        # Pass input through RNN 
        output, hidden_states = self.gru(x, states)

        # Compute logits
        logits = self.out(output)
        
        if return_state:
            return logits, hidden_states
        
        return logits
        

import torch
from torch import nn


class MyBatchNorm1d(nn.Module):
    """
    BatchNorm que acepta (B,T,D) o (B,D).
    OJO: con padding, BN ve los ceros. Es el comportamiento del notebook.
    """
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


class ResLSTMBlock(nn.Module):
    """
    Bloque ResLSTM del notebook:
      x <- BN(x + BiLSTM(x))
      x <- BN(x + BiLSTM(x))
    Cada BiLSTM es num_layers=2, bidirectional=True, hidden_size=d//2 => salida d.
    """
    def __init__(self, d, lstm_layers=2, lstm_dropout=0.1):
        super().__init__()
        assert d % 2 == 0, f"ResLSTM requires even d, got d={d}"

        self.lstm1 = nn.LSTM(
            input_size=d,
            hidden_size=d // 2,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.bn1 = MyBatchNorm1d(d)

        self.lstm2 = nn.LSTM(
            input_size=d,
            hidden_size=d // 2,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.bn2 = MyBatchNorm1d(d)

    def forward(self, x):
        y, _ = self.lstm1(x)
        x = self.bn1(x + y)
        y, _ = self.lstm2(x)
        x = self.bn2(x + y)
        return x


class ResLSTMDecoder(nn.Module):
    """
    Decoder para Brain2Text:
      - day-specific affine (como ya tenías)
      - patching (como ya tenías)
      - proyección a n_units
      - ResLSTMBlock (del notebook)
      - salida a n_classes
    """
    def __init__(
        self,
        neural_dim: int,
        n_units: int,
        n_days: int,
        n_classes: int,
        rnn_dropout: float,
        input_dropout: float,
        n_layers: int,          # no lo usamos aquí, lo dejamos por compat
        patch_size: int,
        patch_stride: int,
        # nuevos knobs (puedes overridearlos):
        reslstm_lstm_layers: int = 2,
        reslstm_lstm_dropout: float = 0.1,
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

        self.day_weights = nn.ParameterList([
            nn.Parameter(torch.eye(neural_dim)) for _ in range(n_days)
        ])
        self.day_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1, neural_dim)) for _ in range(n_days)
        ])

        # Patching => flatten => proj
        in_dim = neural_dim * self.patch_size if self.patch_size > 0 else neural_dim
        self.in_proj = nn.Linear(in_dim, n_units)

        # Notebook ResLSTM
        self.reslstm = ResLSTMBlock(
            d=n_units,
            lstm_layers=reslstm_lstm_layers,
            lstm_dropout=reslstm_lstm_dropout,
        )

        self.dropout = nn.Dropout(p=float(rnn_dropout))
        self.out = nn.Linear(n_units, n_classes)

    def forward(self, features, day_indicies):
        # features: (B,T,C)
        day = int(day_indicies[0].item())
        W = self.day_weights[day]
        b = self.day_biases[day]

        x = self.day_layer_activation(features @ W + b)
        x = self.day_layer_dropout(x)

        # patching
        if self.patch_size > 0:
            ps = self.patch_size
            st = self.patch_stride
            x = x.unfold(dimension=1, size=ps, step=st)     # (B, T', ps, C)
            x = x.contiguous().view(x.size(0), x.size(1), -1)  # (B, T', ps*C)

        x = self.in_proj(x)           # (B, T', n_units)
        x = self.reslstm(x)           # (B, T', n_units)
        x = self.dropout(x)
        logits = self.out(x)          # (B, T', n_classes)
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
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

        if xlstm_num_blocks is None:
            xlstm_num_blocks = int(n_layers)
        if xlstm_dropout is None:
            xlstm_dropout = float(rnn_dropout)

        if n_units % xlstm_num_heads != 0:
            raise ValueError(f"xlstm_num_heads must divide n_units. Got n_units={n_units}, heads={xlstm_num_heads}")

        # Day-specific input layers (mismo patrón que tus modelos)
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList([nn.Parameter(torch.eye(neural_dim)) for _ in range(n_days)])
        self.day_biases = nn.ParameterList([nn.Parameter(torch.zeros(1, neural_dim)) for _ in range(n_days)])
        self.day_layer_dropout = nn.Dropout(p=float(input_dropout))

        # Patching: (B,T,C) -> (B,T', patch_size*C)
        if self.patch_size > 0:
            self.in_proj = nn.Linear(self.patch_size * neural_dim, n_units, bias=True)
        else:
            self.in_proj = nn.Linear(neural_dim, n_units, bias=True)

        # sLSTM-only stack (evita mLSTM y “context_length” dinámico para empezar)
        slstm_layer_cfg = sLSTMLayerConfig(
            embedding_dim=n_units,
            num_heads=int(xlstm_num_heads),
            conv1d_kernel_size=int(xlstm_conv1d_kernel_size),
            dropout=float(xlstm_dropout),
        )
        slstm_block_cfg = sLSTMBlockConfig(slstm=slstm_layer_cfg, feedforward=None)

        stack_cfg = xLSTMBlockStackConfig(
            mlstm_block=None,
            slstm_block=slstm_block_cfg,
            context_length=-1,
            num_blocks=int(xlstm_num_blocks),
            embedding_dim=int(n_units),
            add_post_blocks_norm=True,
            bias=False,
            dropout=float(xlstm_dropout),
            slstm_at="all",
        )
        self.xlstm = xLSTMBlockStack(stack_cfg)

        self.dropout = nn.Dropout(p=float(rnn_dropout))
        self.out = nn.Linear(n_units, n_classes, bias=True)

    def _apply_day_layer(self, x: torch.Tensor, day_indicies: torch.Tensor) -> torch.Tensor:
        # day_indicies: (B,) o (B,1)
        day = int(day_indicies[0].item())
        W = self.day_weights[day]
        b = self.day_biases[day]
        x = x @ W + b
        x = self.day_layer_activation(x)
        x = self.day_layer_dropout(x)
        return x

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        if self.patch_size <= 0:
            return x

        B, T, C = x.shape
        ps = self.patch_size
        st = self.patch_stride
        if st <= 0:
            raise ValueError(f"Invalid patch_stride={st}")

        # unfold temporal
        x = x.unfold(dimension=1, size=ps, step=st)          # (B, T', C, ps)
        x = x.permute(0, 1, 3, 2).contiguous()               # (B, T', ps, C)
        x = x.view(B, x.shape[1], ps * C)                    # (B, T', ps*C)
        return x

    def forward(self, features: torch.Tensor, day_indicies: torch.Tensor) -> torch.Tensor:
        # features: (B,T,C)
        x = self._apply_day_layer(features, day_indicies)
        x = self._patchify(x)
        x = self.in_proj(x)
        x = self.xlstm(x)
        x = self.dropout(x)
        logits = self.out(x)
        return logits
