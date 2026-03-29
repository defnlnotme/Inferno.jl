import torch
from safetensors import safe_open

with safe_open('tests/models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors', framework='pt') as f:
    # Check layer 0 input norm
    layer0_norm = f.get_tensor('model.language_model.layers.0.input_layernorm.weight')
    print(f'Layer 0 input_layernorm weight stats:')
    print(f'  mean: {layer0_norm.mean().item():.4f}')
    print(f'  std: {layer0_norm.std().item():.4f}')
    print(f'  min: {layer0_norm.min().item():.4f}')
    print(f'  max: {layer0_norm.max().item():.4f}')
    
    # Check layer 0 ssm norm
    layer0_ssm_norm = f.get_tensor('model.language_model.layers.0.linear_attn.norm.weight')
    print(f'\nLayer 0 linear_attn.norm weight stats:')
    print(f'  mean: {layer0_ssm_norm.mean().item():.4f}')
    print(f'  std: {layer0_ssm_norm.std().item():.4f}')
    print(f'  min: {layer0_ssm_norm.min().item():.4f}')
    print(f'  max: {layer0_ssm_norm.max().item():.4f}')
