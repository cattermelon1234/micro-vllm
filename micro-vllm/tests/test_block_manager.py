"""BlockManager: allocation, refcounting, prefix cache, and block publishing."""
import random

import pytest

from conftest import (BlockManager, FakeSequence, check_block_invariants,
                      check_prefix_cache_sound)


# --------------------------------------------------------------- allocate ---

def test_allocate_cold_takes_blocks_from_free_list(bm):
    seq = FakeSequence(list(range(9)))          # 9 tokens -> 3 blocks
    assert bm.allocate(seq) == 0                # nothing cached
    assert len(seq.block_table) == seq.num_blocks() == 3
    assert len(bm.free_blocks) == 16 - 3
    assert all(bm.block_list[b].ref_count == 1 for b in seq.block_table)
    assert seq.num_cached_tokens == 0
    check_block_invariants(bm)


def test_allocate_reserves_the_whole_prompt_not_just_a_chunk(bm):
    """Chunking splits computation, never reservation."""
    seq = FakeSequence(list(range(14)))         # 4 blocks
    bm.allocate(seq)
    assert len(seq.block_table) == 4


def test_allocate_refuses_when_short_of_blocks():
    bm = BlockManager(block_size=4, num_blocks=2)
    seq = FakeSequence(list(range(9)))          # needs 3, only 2 exist
    assert bm.can_allocate(seq) == -1
    assert bm.allocate(seq) == -1
    assert seq.block_table == []                # and leaves nothing behind
    assert len(bm.free_blocks) == 2
    check_block_invariants(bm)


def test_allocate_rejects_a_sequence_that_already_holds_blocks(bm):
    seq = FakeSequence(list(range(9)))
    bm.allocate(seq)
    with pytest.raises(AssertionError):
        bm.allocate(seq)


# ------------------------------------------------------------- deallocate ---

def test_deallocate_returns_blocks_to_the_free_list(bm):
    seq = FakeSequence(list(range(9)))
    bm.allocate(seq)
    bm.deallocate(seq)
    assert len(bm.free_blocks) == 16
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    check_block_invariants(bm)


def test_deallocate_with_non_contiguous_block_ids(bm):
    """Regression: deallocate once indexed block_table by block_id."""
    burn = FakeSequence(list(range(20)))        # takes blocks 0..4
    bm.allocate(burn)
    seq = FakeSequence(list(range(100, 109)))   # gets 5, 6, 7
    bm.allocate(seq)
    assert min(seq.block_table) >= 5
    bm.deallocate(seq)                          # used to raise IndexError
    assert set(seq_ids := range(5, 8)) <= set(bm.free_blocks)
    check_block_invariants(bm)


def test_shared_block_survives_until_the_last_owner_releases(bm):
    a = FakeSequence(list(range(12)))
    bm.allocate(a)
    a.num_scheduled_tokens = 12
    bm.publish_blocks(a)
    a.num_cached_tokens = 12

    b = FakeSequence(list(range(12)))           # identical prefix
    assert bm.allocate(b) > 0
    shared = b.block_table[0]
    assert bm.block_list[shared].ref_count == 2

    bm.deallocate(a)
    assert bm.block_list[shared].ref_count == 1
    assert shared not in bm.free_blocks         # b still holds it
    bm.deallocate(b)
    assert bm.block_list[shared].ref_count == 0
    check_block_invariants(bm)


# ----------------------------------------------------------- prefix cache ---

def test_resurrect_a_freed_block_from_the_prefix_cache(bm):
    a = FakeSequence(list(range(9)))
    bm.allocate(a)
    a.num_scheduled_tokens = 9
    bm.publish_blocks(a)
    a.num_cached_tokens = 9
    published = list(a.block_table[:2])
    bm.deallocate(a)                            # refcounts -> 0, back on free list

    b = FakeSequence(list(range(9)))            # same prompt
    assert bm.can_allocate(b) == 2
    assert bm.allocate(b) == 2
    assert b.block_table[:2] == published       # same physical blocks, not copies
    assert b.num_cached_tokens == 8
    for bid in published:
        assert bid not in bm.free_blocks        # pulled back out of the free list
    check_block_invariants(bm)


def test_resurrected_blocks_hold_the_matching_tokens(bm):
    a = FakeSequence(list(range(12)))
    bm.allocate(a)
    a.num_scheduled_tokens = 12
    bm.publish_blocks(a)
    a.num_cached_tokens = 12
    bm.deallocate(a)

    b = FakeSequence(list(range(12)))
    n = bm.allocate(b)
    for i in range(n):
        assert bm.block_list[b.block_table[i]].token_ids == b.block(i)
    check_prefix_cache_sound(bm)


def test_a_different_prefix_does_not_hit(bm):
    a = FakeSequence(list(range(12)))
    bm.allocate(a)
    a.num_scheduled_tokens = 12
    bm.publish_blocks(a)
    a.num_cached_tokens = 12
    bm.deallocate(a)

    b = FakeSequence([99, 98, 97, 96] + list(range(4, 12)))
    assert bm.can_allocate(b) == 0


def test_recycling_a_block_evicts_its_stale_cache_entry():
    bm = BlockManager(block_size=4, num_blocks=3)
    a = FakeSequence(list(range(12)))
    bm.allocate(a)
    a.num_scheduled_tokens = 12
    bm.publish_blocks(a)
    a.num_cached_tokens = 12
    assert len(bm.prefix_cache) == 3
    bm.deallocate(a)

    b = FakeSequence(list(range(100, 112)))     # forces reuse of every block
    bm.allocate(b)
    for h, bid in bm.prefix_cache.items():
        assert bm.block_list[bid].hash == h     # no stale entries left behind
    check_block_invariants(bm)


# --------------------------------------------------------- publish_blocks ---

def test_publish_blocks_skips_a_partial_trailing_block(bm):
    seq = FakeSequence(list(range(9)))          # 2 full blocks + 1 token
    bm.allocate(seq)
    seq.num_scheduled_tokens = 9
    bm.publish_blocks(seq)
    assert len(bm.prefix_cache) == 2
    assert bm.block_list[seq.block_table[2]].hash == -1
    check_prefix_cache_sound(bm)


def test_publish_blocks_is_a_no_op_mid_block(bm):
    seq = FakeSequence(list(range(9)))
    bm.allocate(seq)
    seq.num_cached_tokens, seq.num_scheduled_tokens = 9, 1
    bm.publish_blocks(seq)
    assert bm.prefix_cache == {}


def test_publish_blocks_across_chunked_passes(bm):
    """One pass can complete several blocks; a later pass picks up the chain."""
    seq = FakeSequence(list(range(14)))
    bm.allocate(seq)
    published = []
    for chunk in (6, 6, 2):
        seq.num_scheduled_tokens = chunk
        before = set(bm.prefix_cache.values())
        bm.publish_blocks(seq)
        seq.num_cached_tokens += chunk
        published.append(sorted(set(bm.prefix_cache.values()) - before))
    # pass 1 fills block 0; pass 2 completes blocks 1 and 2; pass 3 leaves 3 partial
    assert [len(p) for p in published] == [1, 2, 0]
    check_prefix_cache_sound(bm)


def test_publish_blocks_chains_hashes_so_prefixes_differ(bm):
    """Same block contents under a different prefix must hash differently."""
    a = FakeSequence([1, 1, 1, 1, 5, 5, 5, 5])
    b = FakeSequence([2, 2, 2, 2, 5, 5, 5, 5])
    for s in (a, b):
        bm.allocate(s)
        s.num_scheduled_tokens = 8
        bm.publish_blocks(s)
        s.num_cached_tokens = 8
    second_of_a = bm.block_list[a.block_table[1]].hash
    second_of_b = bm.block_list[b.block_table[1]].hash
    assert second_of_a != second_of_b


# ---------------------------------------------------------- append / grow ---

def test_can_append_only_needs_a_block_on_a_boundary(bm):
    seq = FakeSequence(list(range(8)))
    bm.allocate(seq)
    seq.num_tokens = 9                          # just crossed into a new block
    assert bm.can_append(seq) is True
    seq.num_tokens = 10
    assert bm.can_append(seq) is True            # needs nothing, trivially true


def test_can_append_false_when_a_block_is_needed_and_none_free():
    bm = BlockManager(block_size=4, num_blocks=2)
    seq = FakeSequence(list(range(8)))
    bm.allocate(seq)                            # consumes both blocks
    seq.num_tokens = 9
    assert bm.can_append(seq) is False


def test_try_append_allocates_exactly_on_the_boundary(bm):
    seq = FakeSequence(list(range(8)))
    bm.allocate(seq)
    assert len(seq.block_table) == 2
    seq.append_token(42)                        # 9 tokens -> needs a 3rd block
    bm.try_append(seq)
    assert len(seq.block_table) == 3 == seq.num_blocks()
    seq.append_token(43)                        # 10 tokens -> still 3 blocks
    bm.try_append(seq)
    assert len(seq.block_table) == 3
    check_block_invariants(bm)


# ------------------------------------------------------------------- soak ---

def test_randomized_allocate_deallocate_conserves_blocks():
    random.seed(0)
    bm = BlockManager(block_size=4, num_blocks=24)
    live = []
    check_block_invariants(bm, "init")
    for step in range(400):
        if live and random.random() < 0.45:
            bm.deallocate(live.pop(random.randrange(len(live))))
        else:
            prompt = [1, 2, 3, 4, 5, 6, 7, 8] + \
                     [random.randrange(50) for _ in range(random.randrange(1, 9))]
            seq = FakeSequence(prompt)
            if bm.allocate(seq) != -1:
                live.append(seq)
        check_block_invariants(bm, f"step {step}")
    for seq in live:
        bm.deallocate(seq)
    check_block_invariants(bm, "drained")
    assert len(bm.free_blocks) == 24
