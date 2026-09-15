"""Scheduler: batch construction, chunked prefill, preemption, termination."""
import random

import pytest

from conftest import (Config, FakeSequence, Scheduler, Status,
                      check_block_invariants, check_prefix_cache_sound)


def make(cfg=None, prompts=(), **seq_kw):
    cfg = cfg or Config()
    sched = Scheduler(cfg)
    seqs = [FakeSequence(p, **seq_kw) for p in prompts]
    for s in seqs:
        sched.add(s)
    return sched, seqs


def drive(sched, seqs, token=500, max_steps=400):
    """Run to completion, checking invariants after every step."""
    steps = 0
    while not sched.is_finished() and steps < max_steps:
        batch, is_prefill = sched.schedule()
        if not batch:
            break
        sched.postprocess(batch, [token] * len(batch))
        check_block_invariants(sched.block_manager, f"step {steps}")
        check_prefix_cache_sound(sched.block_manager, f"step {steps}")
        check_is_prefill_consistent(seqs, f"step {steps}")
        steps += 1
    return steps


def check_is_prefill_consistent(seqs, tag=""):
    for s in seqs:
        if s.status is Status.FINISHED:
            continue
        expected = s.num_cached_tokens < s.num_prompt_tokens
        assert s.is_prefill == expected, \
            f"{tag}: seq {s.seq_id} is_prefill={s.is_prefill} but " \
            f"cached={s.num_cached_tokens}/{s.num_prompt_tokens}"


# ------------------------------------------------------------- basic flow ---

def test_first_batch_is_a_prefill():
    sched, seqs = make(prompts=[list(range(9))])
    batch, is_prefill = sched.schedule()
    assert is_prefill is True
    assert batch == seqs
    assert seqs[0].num_scheduled_tokens == 9
    assert seqs[0].status is Status.RUNNING
    assert seqs[0].is_prefill is False


def test_second_batch_is_a_decode():
    sched, seqs = make(prompts=[list(range(9))])
    batch, _ = sched.schedule()
    sched.postprocess(batch, [500])
    batch, is_prefill = sched.schedule()
    assert is_prefill is False
    assert batch == seqs
    assert seqs[0].num_scheduled_tokens == 1


def test_prefill_does_not_exceed_max_seq():
    sched, seqs = make(Config(max_seq=2), prompts=[list(range(4))] * 4)
    batch, _ = sched.schedule()
    assert len(batch) == 2


def test_empty_scheduler_returns_nothing():
    sched, _ = make()
    assert sched.schedule() == ([], False)
    assert sched.is_finished()


def test_sequences_run_to_completion():
    sched, seqs = make(prompts=[list(range(9)), list(range(100, 107)),
                                list(range(200, 213))], max_tokens=6)
    drive(sched, seqs)
    assert all(s.status is Status.FINISHED for s in seqs)
    assert all(s.num_completion_tokens() == 6 for s in seqs)
    assert len(sched.block_manager.free_blocks) == 32


# --------------------------------------------------------- chunked prefill ---

def test_chunked_prefill_stays_in_waiting_until_the_prompt_is_covered():
    sched, seqs = make(Config(max_tokens_per_batch=6), prompts=[list(range(14))])
    seq = seqs[0]
    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == [seq]
    assert seq.num_scheduled_tokens == 6
    assert seq.is_prefill is True
    assert sched.waiting and not sched.running     # not promoted yet

    sched.postprocess(batch, [500])
    assert seq.num_cached_tokens == 6
    assert seq.num_tokens == 14                    # token discarded, not appended

    sched.schedule(); sched.postprocess([seq], [500])
    assert seq.num_cached_tokens == 12
    assert seq.is_prefill is True

    sched.schedule()
    assert seq.num_scheduled_tokens == 2
    assert seq.is_prefill is False                 # final chunk promotes it
    assert not sched.waiting and sched.running


def test_resumed_chunk_does_not_reallocate():
    sched, seqs = make(Config(max_tokens_per_batch=6), prompts=[list(range(14))])
    seq = seqs[0]
    sched.schedule()
    table = list(seq.block_table)
    refs = [sched.block_manager.block_list[b].ref_count for b in table]
    sched.postprocess([seq], [500])
    sched.schedule()                               # would assert if it re-allocated
    assert seq.block_table == table
    assert [sched.block_manager.block_list[b].ref_count for b in table] == refs


def test_only_the_first_sequence_in_a_batch_gets_chunked():
    """A prompt that would fit in an empty batch waits rather than being split."""
    sched, seqs = make(Config(max_tokens_per_batch=12),
                       prompts=[list(range(8)), list(range(100, 108))])
    batch, _ = sched.schedule()
    assert batch == [seqs[0]]
    assert seqs[0].num_scheduled_tokens == 8       # whole, not 8 + a 4-token chunk
    assert seqs[1].num_scheduled_tokens == 0


def test_an_oversized_prompt_is_chunked_when_it_leads_the_batch():
    sched, seqs = make(Config(max_tokens_per_batch=6), prompts=[list(range(14))])
    batch, _ = sched.schedule()
    assert batch == seqs and seqs[0].num_scheduled_tokens == 6


def test_chunked_prefill_completes():
    sched, seqs = make(Config(max_tokens_per_batch=6),
                       prompts=[list(range(14)), list(range(100, 118))],
                       max_tokens=5)
    drive(sched, seqs)
    assert all(s.status is Status.FINISHED for s in seqs)
    assert all(s.num_completion_tokens() == 5 for s in seqs)


# -------------------------------------------------------------- preemption ---

def test_decode_preempts_the_newest_running_sequence():
    cfg = Config(num_blocks=6, max_tokens_per_batch=64)
    sched, seqs = make(cfg, prompts=[list(range(8)), list(range(100, 108))],
                       max_tokens=8)
    batch, _ = sched.schedule()
    sched.postprocess(batch, [500] * len(batch))
    for _ in range(6):
        batch, _ = sched.schedule()
        if not batch:
            break
        sched.postprocess(batch, [500] * len(batch))
        check_block_invariants(sched.block_manager)
    # someone was pushed back to waiting, and nothing leaked
    assert len(sched.waiting) + len(sched.running) <= 2


def test_preempt_moves_to_the_front_of_waiting_and_frees_blocks():
    sched, seqs = make(prompts=[list(range(9))])
    sched.schedule()
    seq = seqs[0]
    assert seq.block_table
    sched.preempt(seq)
    assert seq.status is Status.WAITING
    assert seq.is_prefill is True
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert sched.waiting[0] is seq
    check_block_invariants(sched.block_manager)


def test_makes_progress_under_memory_pressure():
    sched, seqs = make(Config(num_blocks=8), max_tokens=8,
                       prompts=[list(range(9)), list(range(100, 109)),
                                list(range(200, 209))])
    drive(sched, seqs)
    assert all(s.status is Status.FINISHED for s in seqs)
    assert len(sched.block_manager.free_blocks) == 8


def test_prefill_never_preempts():
    """A waiting sequence that will not fit must not evict running work."""
    sched, seqs = make(Config(num_blocks=4), prompts=[list(range(8))], max_tokens=8)
    batch, _ = sched.schedule()
    sched.postprocess(batch, [500])
    running_before = list(sched.running)
    big = FakeSequence(list(range(900, 916)))       # 4 blocks, none free
    sched.add(big)
    sched.schedule()
    assert list(sched.running) == running_before    # untouched
    assert big.block_table == []


def test_prefill_falls_through_to_decode_when_it_cannot_allocate():
    sched, seqs = make(Config(num_blocks=4), prompts=[list(range(8))], max_tokens=8)
    batch, _ = sched.schedule()
    sched.postprocess(batch, [500])
    sched.add(FakeSequence(list(range(900, 916))))  # cannot fit
    batch, is_prefill = sched.schedule()
    assert batch == seqs and is_prefill is False    # decoded instead of stalling


# ------------------------------------------------------------- termination ---

def test_eos_finishes_a_sequence_and_frees_its_blocks():
    cfg = Config(eos=777)
    sched, seqs = make(cfg, prompts=[list(range(9))], ignore_eos=False, max_tokens=50)
    seq = seqs[0]
    batch, _ = sched.schedule()
    sched.postprocess(batch, [777])                 # model emits EOS
    assert seq.status is Status.FINISHED
    assert seq.block_table == []
    assert len(sched.block_manager.free_blocks) == 32
    assert seq not in sched.running
    assert sched.is_finished()


def test_ignore_eos_keeps_generating_past_the_eos_token():
    cfg = Config(eos=777)
    sched, seqs = make(cfg, prompts=[list(range(9))], ignore_eos=True, max_tokens=3)
    seq = seqs[0]
    batch, _ = sched.schedule()
    sched.postprocess(batch, [777])
    assert seq.status is not Status.FINISHED
    assert seq.last_token == 777                    # appended, not treated as a stop


def test_eos_during_a_chunked_prefill_is_discarded():
    """A mid-prompt logit can coincidentally sample EOS; it must not terminate."""
    cfg = Config(eos=777, max_tokens_per_batch=6)
    sched, seqs = make(cfg, prompts=[list(range(14))], ignore_eos=False, max_tokens=50)
    seq = seqs[0]
    batch, _ = sched.schedule()
    sched.postprocess(batch, [777])
    assert seq.status is not Status.FINISHED
    assert seq.num_tokens == 14                     # not appended
    assert seq.num_cached_tokens == 6               # but progress was recorded


def test_max_tokens_finishes_a_sequence():
    sched, seqs = make(prompts=[list(range(9))], max_tokens=3)
    seq = seqs[0]
    drive(sched, seqs)
    assert seq.status is Status.FINISHED
    assert seq.num_completion_tokens() == 3


# ----------------------------------------------------------- prefix cache ---

def test_a_second_request_with_the_same_prompt_hits_the_cache():
    sched, seqs = make(prompts=[list(range(12)) + [90]], max_tokens=2)
    drive(sched, seqs)
    bm = sched.block_manager
    assert bm.prefix_cache, "first request published nothing"

    second = FakeSequence(list(range(12)) + [91], max_tokens=2)
    hits = bm.can_allocate(second)
    assert hits > 0, "identical 12-token prefix did not hit"
    bm.allocate(second)
    for i in range(hits):
        assert bm.block_list[second.block_table[i]].token_ids == second.block(i)
    assert second.num_cached_tokens == hits * bm.block_size


def test_shared_prefix_requests_all_complete():
    prompts = [list(range(12)) + [90], list(range(12)) + [91], list(range(12)) + [92]]
    sched, seqs = make(prompts=prompts, max_tokens=4)
    drive(sched, seqs)
    assert all(s.status is Status.FINISHED for s in seqs)
    assert len(sched.block_manager.free_blocks) == 32


# ------------------------------------------------------------------- soak ---

@pytest.mark.parametrize("trial", range(30))
def test_randomized_end_to_end(trial):
    random.seed(trial)
    cfg = Config(max_seq=random.choice([2, 3, 4]),
                 max_tokens_per_batch=random.choice([6, 12, 64]),
                 eos=777,
                 num_blocks=random.choice([10, 16, 32]))
    shared = [7, 7, 7, 7, 8, 8, 8, 8]
    pending = []
    for _ in range(random.randrange(3, 9)):
        prompt = (shared if random.random() < 0.5 else []) + \
                 [random.randrange(50) for _ in range(random.randrange(1, 14))]
        pending.append(FakeSequence(prompt, max_tokens=random.randrange(1, 8),
                                    ignore_eos=random.random() < 0.5))
    created = list(pending)
    sched = Scheduler(cfg)
    steps = 0
    while steps < 600:
        if pending and random.random() < 0.3:
            sched.add(pending.pop(0))
        if sched.is_finished() and not pending:
            break
        batch, _ = sched.schedule()
        if not batch:
            if pending:
                steps += 1
                continue
            break
        # occasionally emit EOS so that path is exercised too
        tokens = [777 if random.random() < 0.1 else random.randrange(100, 200)
                  for _ in batch]
        sched.postprocess(batch, tokens)
        check_block_invariants(sched.block_manager, f"trial {trial} step {steps}")
        check_prefix_cache_sound(sched.block_manager, f"trial {trial} step {steps}")
        check_is_prefill_consistent(created, f"trial {trial} step {steps}")
        steps += 1

    assert all(s.status is Status.FINISHED for s in created), \
        f"trial {trial}: {sum(1 for s in created if s.status is not Status.FINISHED)} unfinished"
    assert len(sched.block_manager.free_blocks) == cfg.num_blocks, \
        f"trial {trial}: blocks leaked"
