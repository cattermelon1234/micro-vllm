import torch 
from torch import nn 
import torch.nn.functional as F 
from config import QwenConfig

class RMSNorm(nn.Module):
    def __init__(
            self,
            dim: int,
            eps: float = 1e-6
        ):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim)); # tells pytorch weight is learnable
            self.eps = eps

    def _norm(self, x: torch.Tensor):
        rms_inverse = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps) # dim = -1 because last dim is token dim
        return (x.float() * rms_inverse).to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor
        ):
            normalized = self._norm(x)
            return normalized * self.weight


class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.max_positions = config.max_position_embeddings

    def apply_rotary_emb(self, x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin

    def forward(self, x, position_ids):
        if (position_ids < 0).any() or (position_ids >= self.max_positions).any():
            raise ValueError("position_ids exceed configured context range")
        # Recompute frequencies in float32 so model.bfloat16() cannot round
        # a stored frequency buffer. Only construct the requested positions.
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            inv_freq = 1.0 / (self.theta ** (
                torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32)
                / self.head_dim
            ))
            freqs = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1) @ position_ids[:, None, :].float()
            freqs = freqs.transpose(1, 2)
            angles = torch.cat((freqs, freqs), dim=-1)
            cos = angles.cos().to(x.dtype).unsqueeze(1)
            sin = angles.sin().to(x.dtype).unsqueeze(1)
        return self.apply_rotary_emb(x, cos, sin)


class SelfAttention(nn.Module):
    def __init__(
        self, config, layer_idx
        ): 
        super().__init__();
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads # because attn heads != k_heads/q_heads in GQA
        self.head_dim = config.head_dim # head dim remains const even in GQA
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // self.num_kv_heads # number of kv heads per q head

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rope = RoPE(config)

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_attention_heads,
            bias=config.attention_bias,
        )

        # GQA uses fewer K/V heads than query heads, with the same head_dim.
        # Neither combined attention width must equal hidden_size.
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_kv_heads, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_kv_heads, bias=config.attention_bias)

        # input to out proj is the concatenation of all attn heads (num_heads * head_dim -> hidden_size)
        self.proj_out = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states : torch.Tensor,
        position_ids,
        attention_mask: torch.Tensor,
        cache = None,
    ):
        
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # new k, v to append to the k/v cache
        q = self.q_norm(q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, position_ids) 
        k = self.rope(k, position_ids) 

        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)

        k = k.repeat_interleave(self.num_kv_groups, dim=1) # repeats k, v heads since multiple q heads map to one k/v in GQA
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # transpose last 2 dims 
        attn_scores = attn_scores * (self.head_dim ** -0.5) # scale by 1/sqrt(head_dim)

        attn_scores = attn_scores + attention_mask 
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v)

        # merge query heads and project
        attn_output = attn_output.transpose(1, 2).contiguous() 
        attn_output = attn_output.view(
            batch_size,
            seq_len,
            self.num_attention_heads * self.head_dim
        )

        # [B, query_len, hidden_dim]
        attn_output = self.proj_out(attn_output)
        return attn_output

class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
    def forward(self, x):
            # [B, query_len, hidden_dim]
            return self.down_proj(
                F.silu(self.gate_proj(x)) * self.up_proj(x)
            )

class DynamicCache(nn.Module):
    def __init__(self, config, batch_size, device, dtype):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.seq_len = 0
        empty_shape = (
            batch_size,
            config.num_key_value_heads,
            0,
            config.head_dim,
        )

        for layer_idx in range(config.num_hidden_layers):
            self.register_buffer(
                f"k_cache_{layer_idx}",
                torch.empty(empty_shape, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"v_cache_{layer_idx}",
                torch.empty(empty_shape, device=device, dtype=dtype),
                persistent=False,
            )

    @property
    def k_cache(self):
        return [
            getattr(self, f"k_cache_{layer_idx}")
            for layer_idx in range(self.config.num_hidden_layers)
        ]

    @property
    def v_cache(self):
        return [
            getattr(self, f"v_cache_{layer_idx}")
            for layer_idx in range(self.config.num_hidden_layers)
        ]

    def update(self, k, v, layer_idx):
        # k/v: [B, Hkv, new_seq_len, D]
        cached_k = getattr(self, f"k_cache_{layer_idx}")
        cached_v = getattr(self, f"v_cache_{layer_idx}")

        updated_k = torch.cat((cached_k, k), dim=2)
        updated_v = torch.cat((cached_v, v), dim=2)
        setattr(self, f"k_cache_{layer_idx}", updated_k)
        setattr(self, f"v_cache_{layer_idx}", updated_v)

        # only increment seq len on first layer to prevent overcounting
        if layer_idx == 0:
            self.seq_len += k.shape[2]

        return updated_k, updated_v

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__();
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = SelfAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.mlp = QwenMLP(config)

    def forward(self, hidden_states, position_ids, attention_mask, cache=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.attention(hidden_states, position_ids, attention_mask, cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        # [B, query_len, D]
        return hidden_states

def make_attention_mask(
      query_len,
      past_len,
      padding_mask,  # [B, K]
      dtype,
      device,
  ):
      kv_len = past_len + query_len

      # Physical positions in the KV cache.
      query_positions = (
          past_len + torch.arange(query_len, device=device)
      )  # [Q]

      key_positions = torch.arange(kv_len, device=device)  # [K]

      # [1, Q, K]
      causal_allowed = (
          key_positions[None, None, :]
          <= query_positions[None, :, None]
      )

      # [B, 1, K]
      key_is_real = padding_mask[:, None, :kv_len].bool()

      # [B, Q, K]
      allowed = causal_allowed & key_is_real

      mask = torch.zeros(
          allowed.shape,
          dtype=dtype,
          device=device,
      )
      mask.masked_fill_(~allowed, torch.finfo(dtype).min)

      # [B, 1, Q, K]
      return mask.unsqueeze(1)

class Qwen(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        cache=None,
    ):
        # [B, S] -> [B, S, hidden_size]
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids,
                attention_mask,
                cache,
            )

        hidden_states = self.norm(hidden_states)

        # [B, query_len, hidden_size] -> [B, query_len, vocab_size]
        logits = self.lm_head(hidden_states)

        return logits

    def generate(self, input_ids, max_new_tokens, padding_mask=None):
        self.eval()

        if max_new_tokens <= 0:
            return input_ids

        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain a nonempty prompt")
        batch_size, query_len = input_ids.shape
        if padding_mask is None:
            padding_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            padding_mask = padding_mask.bool()

        if padding_mask.shape != input_ids.shape:
            raise ValueError("padding_mask must match input_ids")
        if not padding_mask[:, -1].all() or (padding_mask[:, :-1] & ~padding_mask[:, 1:]).any():
            raise ValueError("Use left padding with at least one real token per prompt")
        if query_len + max_new_tokens - 1 > self.config.max_position_embeddings:
            raise ValueError("Generation exceeds configured context range")

        cache = DynamicCache(
            config=self.config,
            batch_size=batch_size,
            device=input_ids.device,
            dtype=self.embed_tokens.weight.dtype,
        )
        generated = input_ids

        with torch.inference_mode():
            position_ids = padding_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(~padding_mask, 0)
            attention_mask = make_attention_mask(
                query_len=query_len,
                past_len=0,
                padding_mask=padding_mask,
                dtype=self.embed_tokens.weight.dtype,
                device=input_ids.device,
            )
            logits = self(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache=cache,
            )

            finished = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=input_ids.device,
            )

            for step in range(max_new_tokens):
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

                # torch.where(condition, A, B) replaces with A if cond is true at idx,
                # replaces with B if cond is false.
                # finished is [[true], [true], [false], ...] (batch_size x 1)
                # next_token is [[batch_1_token], [batch_2_token], ...] (batch_size x 1)
                # torch.full_like [[eos_token], [eos_token], [eos_token], ...]
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, self.config.pad_token_id),
                    next_token,
                )

                generated = torch.cat((generated, next_token), dim=1)
                padding_mask = torch.cat(
                    (padding_mask, ~finished.unsqueeze(1)),
                    dim=1,
                )

                finished = finished | (
                    next_token.squeeze(1) == self.config.eos_token_id
                )

                if finished.all() or step == max_new_tokens - 1:
                    break

                past_len = cache.seq_len
                position_ids = padding_mask.long().sum(dim=-1, keepdim=True) - 1
                attention_mask = make_attention_mask(
                    query_len=1,
                    past_len=past_len,
                    padding_mask=padding_mask,
                    dtype=self.embed_tokens.weight.dtype,
                    device=input_ids.device,
                )
                logits = self(
                    input_ids=next_token,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    cache=cache,
                )

        return generated

if __name__ == "__main__":
    config = QwenConfig() 
    batch_size = 4
    seq_len = 24
    norm = RMSNorm(64)
    out = norm(torch.randn(4, 20, 64))
    print(out.shape)

    rope = RoPE(config)
    query = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)
    print("query: ")
    print(query.shape)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    rotated_query = rope(query, position_ids)
    print(rotated_query.shape)
