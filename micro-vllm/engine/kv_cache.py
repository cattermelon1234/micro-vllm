"""Fixed physical KV storage. Block ownership stays in BlockManager."""
import torch


class PagedKVCache:
    def __init__(self, model_config, *, num_blocks, block_size, device, dtype):
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (
            model_config.num_hidden_layers,
            num_blocks,
            block_size,
            model_config.num_key_value_heads,
            model_config.head_dim,
        )
        # nano-vLLM layout: one allocation, with K/V in the leading dimension.
        # Each layer receives views into this storage below.
        self.tensor = torch.empty((2, *shape), device=device, dtype=dtype)
        self.k = self.tensor[0]
        self.v = self.tensor[1]
        self.layers = None

    def bind(self, model):
        """Attach each layer's K/V cache views to its attention module."""
        layers = getattr(model, "layers", None)
        if layers is None or len(layers) != self.tensor.shape[1]:
            raise ValueError("Model layers do not match the KV cache configuration")
        for layer_idx, layer in enumerate(layers):
            attention = layer.attention
            attention.k_cache = self.tensor[0, layer_idx]
            attention.v_cache = self.tensor[1, layer_idx]
        self.layers = layers

    def write(self, layer_idx, slot_mapping, keys, values):
        """keys/values: [new_tokens, kv_heads, head_dim]; slots: [new_tokens]."""
        # A slot is physical_block * block_size + offset_inside_block.
        # Flattening these two dimensions gives indexed writes without reallocating.
        if self.layers is None:
            raise RuntimeError("Bind the KV cache to the model before writing")
        attention = self.layers[layer_idx].attention
        k_slots = attention.k_cache.flatten(0, 1)
        v_slots = attention.v_cache.flatten(0, 1)
        k_slots.index_copy_(0, slot_mapping, keys)
        v_slots.index_copy_(0, slot_mapping, values)

    def gather(self, layer_idx, block_table, context_len):
        """Return valid K/V in logical order: [1, kv_heads, context_len, head_dim]."""
        num_blocks = (context_len + self.block_size - 1) // self.block_size
        blocks = block_table[:num_blocks]

        def gather_tensor(pool):
            tokens = pool[layer_idx].index_select(0, blocks).flatten(0, 1)
            return tokens[:context_len].transpose(0, 1).unsqueeze(0)

        # This copies into contiguous logical token order. A future paged-attention
        # kernel should read the block table directly and avoid these gathers.
        return gather_tensor(self.k), gather_tensor(self.v)


class SequenceCache:
    """Adapter for Qwen's existing cache.update(k, v, layer_idx) interface."""

    def __init__(self, pool, block_table, positions):
        self.pool = pool
        self.block_table = block_table
        self.context_len = int(positions[-1].item()) + 1
        blocks = block_table[positions // pool.block_size]
        self.slot_mapping = blocks * pool.block_size + positions % pool.block_size

    def update(self, k, v, layer_idx):
        # Qwen supplies [1, kv_heads, new_tokens, head_dim]. Cache storage uses
        # [new_tokens, kv_heads, head_dim]. K has already received RoPE here.
        self.pool.write(
            layer_idx, self.slot_mapping,
            k[0].transpose(0, 1), v[0].transpose(0, 1),
        )
        return self.pool.gather(layer_idx, self.block_table, self.context_len)
