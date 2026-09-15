"""Shared fixtures for the engine tests.

`engine/sequence.py` does not import yet (`sampling_params` is referenced in
`__init__` but never passed in, and `block()` slices with
`(start + block_size) % max_tokens`). Until that is fixed these tests run
against the `FakeSequence` shim below, which implements the same surface with
correct block slicing.

To switch over once sequence.py works: delete FakeSequence and import the real
Sequence/Status here. The only differences are the constructor signature
(sampling_params) and block_size, which the fixtures set explicitly.
"""
import sys
from enum import Enum, auto
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(ENGINE))


class Status(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class FakeSequence:
    block_size = 4          # small, so block arithmetic is checkable by hand
    _next_id = 0

    def __init__(self, token_ids, max_tokens=8, ignore_eos=True, temperature=0.0):
        FakeSequence._next_id += 1
        self.seq_id = FakeSequence._next_id
        self.status = Status.WAITING
        self.token_ids = list(token_ids)
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_scheduled_tokens = 0
        self.num_cached_tokens = 0
        self.block_table = []
        self.is_prefill = True
        self.last_token = token_ids[-1]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos

    def __len__(self):
        return len(self.token_ids)

    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, i):
        start = i * self.block_size
        return self.token_ids[start:start + self.block_size]

    def is_finished(self):
        return self.status == Status.FINISHED

    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.num_tokens += 1
        self.last_token = token_id


def _install_stub():
    """Put FakeSequence in place of engine.sequence before the engine imports it."""
    import types
    mod = types.ModuleType("sequence")
    mod.Sequence = FakeSequence
    mod.Status = Status
    sys.modules["sequence"] = mod
    import block_manager, scheduler
    block_manager.Sequence = FakeSequence
    scheduler.Sequence = FakeSequence
    scheduler.Status = Status
    return block_manager, scheduler


block_manager, scheduler = _install_stub()
BlockManager = block_manager.BlockManager
Block = block_manager.Block
Scheduler = scheduler.Scheduler


class Config:
    """Minimal stand-in for the engine Config the Scheduler expects."""
    def __init__(self, max_seq=4, max_tokens_per_batch=64, eos=999,
                 block_size=4, num_blocks=32):
        self.max_seq = max_seq
        self.max_tokens_per_batch = max_tokens_per_batch
        self.eos = eos
        self.block_size = block_size
        self.num_blocks = num_blocks


@pytest.fixture
def bm():
    return BlockManager(block_size=4, num_blocks=16)


@pytest.fixture
def seq_factory():
    def make(tokens, **kw):
        return FakeSequence(tokens, **kw)
    return make


def check_block_invariants(bm, tag=""):
    """Every block is in exactly one of: the free deque, or held (ref_count > 0)."""
    free = set(bm.free_blocks)
    assert len(free) == len(bm.free_blocks), f"{tag}: duplicate id in free list"
    held = {b.block_id for b in bm.block_list if b.ref_count > 0}
    assert not (free & held), f"{tag}: block both free and held: {free & held}"
    assert free | held == set(range(bm.num_blocks)), \
        f"{tag}: leaked {set(range(bm.num_blocks)) - (free | held)}"
    assert all(b.ref_count >= 0 for b in bm.block_list), f"{tag}: negative refcount"


def check_prefix_cache_sound(bm, tag=""):
    """Every cache entry must point at a block whose contents actually match."""
    for h, block_id in bm.prefix_cache.items():
        block = bm.block_list[block_id]
        assert block.hash == h, \
            f"{tag}: prefix_cache[{h}] -> block {block_id} whose hash is {block.hash}"
        assert len(block.token_ids) == bm.block_size, \
            f"{tag}: published block {block_id} holds {len(block.token_ids)} tokens, " \
            f"expected a full block of {bm.block_size}"
