"""Fused Triton store for new K/V rows in the physical cache."""

try:
    import triton
    import triton.language as tl
except ImportError:  # Keep the CPU reference usable without Triton installed.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _store_kv_kernel(
        key_ptr, value_ptr, key_cache_ptr, value_cache_ptr, slot_ptr,
        key_stride_token, key_stride_head, value_stride_token, value_stride_head,
        KV_WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        slot = tl.load(slot_ptr + token)
        if slot >= 0:
            x = tl.arange(0, BLOCK)
            mask = x < KV_WIDTH
            k = tl.load(
                key_ptr + token * key_stride_token + x,
                mask=mask, other=0,
            )
            v = tl.load(
                value_ptr + token * value_stride_token + x,
                mask=mask, other=0,
            )
            destination = slot * KV_WIDTH + x
            tl.store(key_cache_ptr + destination, k, mask=mask)
            tl.store(value_cache_ptr + destination, v, mask=mask)


def store_kv(keys, values, key_cache, value_cache, slot_mapping):
    """Store [tokens, kv_heads, head_dim] rows at flattened physical slots."""
    if triton is None:
        raise RuntimeError("Triton is required to write CUDA KV cache tensors")
    if not keys.is_cuda or not values.is_cuda or not key_cache.is_cuda:
        raise ValueError("Triton KV stores require CUDA tensors")
    if keys.shape != values.shape or keys.ndim != 3:
        raise ValueError("K and V must have matching [tokens, heads, head_dim] shapes")
    if slot_mapping.numel() != keys.shape[0]:
        raise ValueError("There must be one cache slot per input token")
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError("Layer cache views must have [blocks, block_size, heads, dim] shape")
    if not keys.is_contiguous() or not values.is_contiguous():
        keys = keys.contiguous()
        values = values.contiguous()
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise ValueError("Layer cache views must be contiguous")

    tokens, kv_heads, head_dim = keys.shape
    kv_width = kv_heads * head_dim
    if key_cache.shape[-2:] != (kv_heads, head_dim):
        raise ValueError("Input K/V dimensions do not match the cache")
    if tokens == 0:
        return

    _store_kv_kernel[(tokens,)](
        keys, values, key_cache, value_cache, slot_mapping,
        kv_width, kv_width,
        num_warps=4,
    )
