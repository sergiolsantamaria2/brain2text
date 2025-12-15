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


class ResLSTMDecoder(nn.Module):
    """
    Decoder con LSTM apiladas + residual connections + LayerNorm.
    Diseñado para ser comparable con GRUDecoder y no romper el trainer.

    Notas:
    - Proyecta la entrada (posible patching) a n_units para que el residual sea válido.
    - states (si se usa) debe ser un tuple (h, c) con shape (n_layers, B, n_units).
    """
    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=5,
        patch_size=0,
        patch_stride=0,
        norm_type="layernorm",     # "layernorm" | "none"
        post_norm=False,
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Day-specific layers (idéntico a GRUDecoder)
        self.day_layer_activation = nn.Softsign()

        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        # Input size efectivo (por patching)
        self.input_size = self.neural_dim
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # Proyección a d_model = n_units para residuals
        self.in_proj = nn.Identity() if self.input_size == self.n_units else nn.Linear(self.input_size, self.n_units)

        # Norms
        def make_norm():
            if norm_type == "layernorm":
                return nn.LayerNorm(self.n_units)
            if norm_type == "none":
                return nn.Identity()
            raise ValueError(f"norm_type inválido: {norm_type}")

        self.pre_norms = nn.ModuleList([make_norm() for _ in range(self.n_layers)])
        self.post_norms = nn.ModuleList([make_norm() for _ in range(self.n_layers)]) if post_norm else None

        self.dropout = nn.Dropout(self.rnn_dropout)

        # LSTM por capa (para controlar residual por-layer)
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=self.n_units,
                hidden_size=self.n_units,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            for _ in range(self.n_layers)
        ])

        # Init similar a tu GRU (ortogonal/xavier)
        for lstm in self.lstms:
            for name, param in lstm.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial states (h0, c0)
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.n_layers, 1, self.n_units)))
        self.c0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.n_layers, 1, self.n_units)))

    def forward(self, x, day_idx, states=None, return_state=False):
        # Day-specific projection
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # Optional patching (idéntico a GRUDecoder)
        if self.patch_size > 0:
            x = x.unsqueeze(1)                      # [B, 1, T, D]
            x = x.permute(0, 3, 1, 2)               # [B, D, 1, T]
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2)          # [B, D, N, P]
            x_unfold = x_unfold.permute(0, 2, 3, 1) # [B, N, P, D]
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)  # [B, N, D*P]

        # Project to n_units for residual LSTM stack
        x = self.in_proj(x)

        B = x.shape[0]
        if states is None:
            h = self.h0.expand(self.n_layers, B, self.n_units).contiguous()
            c = self.c0.expand(self.n_layers, B, self.n_units).contiguous()
        else:
            h, c = states

        new_h = []
        new_c = []

        # Residual LSTM stack
        for i in range(self.n_layers):
            residual = x
            x_norm = self.pre_norms[i](x)

            out, (h_i, c_i) = self.lstms[i](x_norm, (h[i:i+1], c[i:i+1]))
            out = self.dropout(out)

            x = residual + out
            if self.post_norms is not None:
                x = self.post_norms[i](x)

            new_h.append(h_i)
            new_c.append(c_i)

        logits = self.out(x)

        if return_state:
            h_new = torch.cat(new_h, dim=0)  # (n_layers, B, n_units)
            c_new = torch.cat(new_c, dim=0)
            return logits, (h_new, c_new)

        return logits
