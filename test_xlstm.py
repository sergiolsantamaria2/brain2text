import sys
sys.path.insert(0, "src")

import torch
from brain2text.model_training.rnn_model import XLSTMDecoder

model = XLSTMDecoder(
    neural_dim=512,
    n_units=768,
    n_days=10,
    n_classes=41,
    rnn_dropout=0.2,
    input_dropout=0.2,
    n_layers=5,
    patch_size=14,
    patch_stride=4,
    xlstm_num_blocks=5,
    xlstm_num_heads=4,
    xlstm_backend="vanilla",  # Forzar vanilla
)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

x = torch.randn(2, 100, 512)
day_idx = torch.tensor([0, 1])
out = model(x, day_idx)
print(f'Output shape: {out.shape}')
print('✓ xLSTM smoke test passed (vanilla backend)')
