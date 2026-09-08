from dataclasses import dataclass, fields


@dataclass
class QwenConfig:
    vocab_size: int = 151_936
    hidden_size: int = 1_024
    intermediate_size: int = 3_072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    # Qwen3's attention width need not equal its residual hidden_size.
    head_dim: int = 128
    max_position_embeddings: int = 40_960
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    bos_token_id: int = 151_643
    eos_token_id: int = 151_645
    pad_token_id: int = 151_643

    def __post_init__(self):
        for name in ('vocab_size', 'hidden_size', 'intermediate_size',
                     'num_hidden_layers', 'num_attention_heads',
                     'num_key_value_heads', 'head_dim', 'max_position_embeddings'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError('num_attention_heads must be divisible by num_key_value_heads')
        if self.head_dim % 2:
            raise ValueError('head_dim must be even for RoPE')

    @classmethod
    def from_hf(cls, config):
        """Read dimensions from the checkpoint, rejecting unsupported architectures."""
        if config.model_type != 'qwen3':
            raise ValueError('Only dense Qwen3 checkpoints are supported')
        if getattr(config, 'rope_scaling', None):
            raise ValueError('Scaled RoPE is not implemented')
        if getattr(config, 'use_sliding_window', False):
            raise ValueError('Sliding-window attention is not implemented')
        if config.hidden_act != 'silu':
            raise ValueError('Only SiLU MLP activation is supported')
        values = {f.name: getattr(config, f.name) for f in fields(cls)
                  if getattr(config, f.name, None) is not None}
        return cls(**values)
