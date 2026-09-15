# Engine tests

```sh
# from this directory
PYTHONPATH=../../qwen/.venv/lib/python3.14/site-packages \
  ../../qwen/.venv/bin/python -m pytest -q
```

The `-m pytest` + `PYTHONPATH` dance is needed because every script in
`qwen/.venv/bin/` still has a shebang pointing at `/Users/brian/Documents/qwen`,
the path the project had before it was renamed to `micro-vllm`. Recreating the
venv (`uv venv` + `uv pip install -r qwen/requirements.txt`) fixes it, after
which plain `pytest -q` works.

## Layout

| file | covers |
|---|---|
| `conftest.py` | `FakeSequence` shim, `Config` stub, invariant helpers |
| `test_block_manager.py` | allocation, refcounting, prefix cache, block publishing |
| `test_scheduler.py` | batch construction, chunked prefill, preemption, termination |

## The FakeSequence shim

`engine/sequence.py` does not import — `__init__` references `sampling_params`,
which is never a parameter, and `block()` slices with
`(start + block_size) % max_tokens` instead of a plain clamped slice. Until both
are fixed, `conftest.FakeSequence` stands in with the same surface and correct
block arithmetic, so these tests exercise the block manager and scheduler rather
than `Sequence` itself.

To switch over: delete `FakeSequence` and import the real `Sequence`/`Status` in
`conftest.py`. The fixtures already set `block_size` explicitly; the only other
difference is the constructor taking sampling params.

## Invariants checked after every step

- no duplicate ids in `free_blocks`
- no block both free and held (`ref_count > 0`)
- `free ∪ held == all blocks` — nothing leaked
- no negative refcounts
- every `prefix_cache` entry points at a block whose hash matches and which is
  completely full
- `seq.is_prefill == (num_cached_tokens < num_prompt_tokens)` for live sequences
