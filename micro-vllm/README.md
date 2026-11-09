# PyTorch paged KV reference

This reference connects the scheduler and block manager to the existing model
in `../qwen`. It runs on the model's device (CPU by default); it does not download
weights. The example uses a tiny randomly initialized Qwen, so its token IDs
demonstrate execution rather than meaningful text generation.

From this directory, using the sibling Qwen virtual environment:

```sh
PYTHONPATH=../qwen:. ../qwen/.venv/bin/python examples/pytorch_reference.py
PYTHONPATH=../qwen:. ../qwen/.venv/bin/python -m pytest -q tests
```

## Physical storage and addresses

`ModelRunner` allocates a fixed `PagedKVCache` once, using the configured block
count. K and V each have shape:

```text
[num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

The prefix index contains `prefix_hash -> physical_block_id`. A sequence's
`block_table` contains `logical_block_index -> physical_block_id`. Neither
contains K/V tensors or raw pointers. For a token at absolute position `p`:

```python
block_id = seq.block_table[p // block_size]
offset = p % block_size
# The actual K/V vectors live at:
k_cache[layer_idx, block_id, offset, :, :]
v_cache[layer_idx, block_id, offset, :, :]
```

PyTorch resolves tensor indices to memory addresses. One block ID reserves
the corresponding token slots across all layers. `PagedKVCache.write` flattens
the block and offset axes so `index_copy_` can write newly computed K/V.

Releasing a block changes ownership in `BlockManager`; it does not free or clear
GPU storage. Full computed blocks remain indexed for prefix reuse. Reassigning
a physical block invalidates its old prefix entry. Only valid token positions
may be read; the unused tail of a block contains uninitialized or stale data.

## Forward execution

`ModelRunner.forward(seq)` processes exactly the token slice selected by
`num_cached_tokens` and `num_scheduled_tokens`. It builds absolute RoPE positions,
the physical slot mapping, and an offset-aware causal mask. A `SequenceCache`
adapter implements Qwen's existing `cache.update(k, v, layer_idx)` interface:

1. Write new K/V into the physical pool after Q/K normalization and RoPE.
2. Gather valid K/V through the block table into logical token order.
3. Return them to Qwen's existing attention implementation.

The same path handles prefill, chunked prefill, prefix hits, and decode. Requests
run separately within a scheduled batch. `run(seqs, is_prefill)` returns one
token ID per request in scheduler order. Intermediate prefill chunks return a
placeholder that the scheduler discards. Completed prefills and decode steps use
greedy or temperature sampling.

To use loaded weights instead, pass the existing `Qwen` instance into
`ModelRunner(model, config)` and provide its tokenizer to `LLMEngine`. Set
`config.eos` to the tokenizer/model EOS ID. Move/cast the model before creating
the runner; this fixes the cache's device and dtype. Use the same engine config
for both runner and engine.

This version uses an explicit memory budget:

```text
KV bytes = 2 * num_layers * num_blocks * block_size
             * num_kv_heads * head_dim * bytes_per_element
```

It has no memory profiling, packed batch attention, Triton kernels, tensor
parallelism, or CUDA graphs. The copies in `PagedKVCache.gather` are intentional
reference behavior. Future paged attention should read physical blocks directly;
a Triton cache-write kernel can replace `PagedKVCache.write` independently.
