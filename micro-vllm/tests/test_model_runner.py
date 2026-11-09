"""Compare paged execution with full-context, uncached Qwen inference.

Run from micro-vllm with PYTHONPATH=../qwen:. so the sibling model is importable.
"""
import pytest
import torch

from config import QwenConfig
from qwen import Qwen

from engine.engine import Config, LLMEngine
from engine.model_runner import ModelRunner
from engine.sampling_params import SamplingParams
from engine.sequence import Sequence


@pytest.fixture
def model():
    torch.manual_seed(17)
    # Attention width intentionally differs from residual hidden width.
    return Qwen(QwenConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        tie_word_embeddings=False, bos_token_id=0, eos_token_id=31, pad_token_id=0,
    )).eval()


@torch.inference_mode()
def reference_logits(model, tokens):
    """Recompute the full sequence, without any KV cache or block tables."""
    ids = torch.tensor([tokens])
    positions = torch.arange(len(tokens))[None, :]
    mask = torch.full((len(tokens), len(tokens)), float("-inf")).triu(1)
    return model(ids, positions, mask[None, None, :, :])[0]


def reference_completion(model, prompt, count):
    tokens = list(prompt)
    for _ in range(count):
        tokens.append(reference_logits(model, tokens)[-1].argmax().item())
    return tokens[len(prompt):]


def drain(engine):
    outputs = {}
    for _ in range(100):
        if engine.is_finished():
            return [outputs[key] for key in sorted(outputs)]
        outputs.update(engine.step())
    pytest.fail("Engine did not finish in 100 steps")


def test_cache_writes_and_gathers_noncontiguous_blocks(model):
    config = Config(block_size=4, num_blocks=6)
    cache = ModelRunner(model, config).kv_cache
    cache.k.fill_(float("nan"))
    cache.v.fill_(float("nan"))
    addresses = (cache.k.data_ptr(), cache.v.data_ptr())
    keys = torch.randn(5, 2, 8)
    values = torch.randn_like(keys)
    # Logical positions 0..4 live in physical blocks 3 and 0.
    cache.write(1, torch.tensor([12, 13, 14, 15, 0]), keys, values)
    k, v = cache.gather(1, torch.tensor([3, 0]), context_len=5)
    torch.testing.assert_close(k, keys.transpose(0, 1)[None, ...])
    torch.testing.assert_close(v, values.transpose(0, 1)[None, ...])
    assert cache.k[0].isnan().all()  # Another layer was not touched.
    assert cache.k[1, 0, 1:].isnan().all()  # Unwritten slots were not read.
    assert addresses == (cache.k.data_ptr(), cache.v.data_ptr())


def test_chunked_prefill_and_decode_logits_match_full_context(model):
    config = Config(block_size=4, num_blocks=6)
    Sequence.block_size = config.block_size
    runner = ModelRunner(model, config)
    runner.kv_cache.k.fill_(float("nan"))
    runner.kv_cache.v.fill_(float("nan"))
    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9], SamplingParams())
    seq.block_table = [5, 1, 3]
    expected = reference_logits(model, seq.token_ids)

    for start, end in [(0, 3), (3, 6), (6, 9)]:
        seq.num_cached_tokens = start
        seq.num_scheduled_tokens = end - start
        torch.testing.assert_close(
            runner.forward(seq), expected[start:end], atol=2e-6, rtol=2e-5,
        )

    seq.num_cached_tokens = 9
    seq.append_token(10)
    seq.num_scheduled_tokens = 1
    torch.testing.assert_close(
        runner.forward(seq), reference_logits(model, seq.token_ids)[-1:],
        atol=2e-6, rtol=2e-5,
    )


@pytest.mark.parametrize("block_size,num_blocks,budget,prompts", [
    (4, 16, 64, [[1, 2, 3, 4, 5], [6, 7], [8]]),
    (4, 16, 2, [[1, 2, 3, 4, 5], [6, 7], [8]]),
    # Two 3-token prompts fill this pool. Decode must preempt and recompute.
    (2, 4, 64, [[1, 2, 3], [4, 5, 6]]),
    (1, 16, 64, [[1, 2, 3], [4, 5, 6]]),
])
def test_engine_matches_uncached_generation(model, monkeypatch, block_size, num_blocks, budget, prompts):
    config = Config(block_size=block_size, num_blocks=num_blocks,
                    max_seq=3, max_tokens_per_batch=budget)
    runner = ModelRunner(model, config)
    engine = LLMEngine(config, runner)
    original_run = runner.run

    def checked_run(seqs, is_prefill):
        assert sum(s.num_scheduled_tokens for s in seqs) <= budget
        assert len({s.seq_id for s in seqs}) == len(seqs)
        return original_run(seqs, is_prefill)

    monkeypatch.setattr(runner, "run", checked_run)
    for prompt in prompts:
        engine.add_request(prompt, SamplingParams(max_tokens=4, ignore_eos=True))
    assert drain(engine) == [reference_completion(model, prompt, 4) for prompt in prompts]
    assert len(engine.scheduler.block_manager.free_blocks) == num_blocks


def test_reuse_live_and_released_prefix_blocks(model):
    config = Config(block_size=2, num_blocks=16, max_tokens_per_batch=64)
    runner = ModelRunner(model, config)
    engine = LLMEngine(config, runner)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    prompt_a, prompt_b, prompt_c = [1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 4, 7]
    engine.add_request(prompt_a, params)
    seq_a = engine.scheduler.waiting[0]
    engine.step()  # Publish two full prefix blocks while A is still running.
    prefix_ids = seq_a.block_table[:2]
    prefix_k = runner.kv_cache.k[:, prefix_ids].clone()
    prefix_v = runner.kv_cache.v[:, prefix_ids].clone()

    engine.add_request(prompt_b, params)
    seq_b = engine.scheduler.waiting[0]
    seqs, is_prefill = engine.scheduler.schedule()
    assert seqs == [seq_b]
    assert seq_b.num_cached_tokens == 4
    assert seq_b.block_table[:2] == prefix_ids
    assert all(engine.scheduler.block_manager.block_list[b].ref_count == 2 for b in prefix_ids)
    engine.scheduler.postprocess(seqs, runner.run(seqs, is_prefill))
    torch.testing.assert_close(runner.kv_cache.k[:, prefix_ids], prefix_k)
    torch.testing.assert_close(runner.kv_cache.v[:, prefix_ids], prefix_v)
    assert drain(engine) == [reference_completion(model, p, 3) for p in (prompt_a, prompt_b)]

    # Released blocks still contain K/V and can be resurrected from the index.
    engine.add_request(prompt_c, params)
    seqs, is_prefill = engine.scheduler.schedule()
    seq_c = seqs[0]
    assert seq_c.num_cached_tokens == 4
    assert seq_c.block_table[:2] == prefix_ids
    engine.scheduler.postprocess(seqs, runner.run(seqs, is_prefill))
    assert drain(engine) == [reference_completion(model, prompt_c, 3)]


def test_eos_and_string_prompt(model):
    prompt = [1, 2, 3]
    first_token = reference_completion(model, prompt, 1)[0]
    config = Config(block_size=2, num_blocks=8, eos=first_token)

    class Tokenizer:
        def encode(self, text):
            assert text == "hello"
            return prompt

        def decode(self, tokens):
            return str(tokens)

    engine = LLMEngine(config, ModelRunner(model, config), Tokenizer())
    assert engine.generate(["hello"], SamplingParams(max_tokens=4)) == [str([first_token])]


def test_sampling_skips_intermediate_prefill(model):
    config = Config(block_size=2, num_blocks=8, max_tokens_per_batch=2)
    runner = ModelRunner(model, config)
    engine = LLMEngine(config, runner)
    engine.add_request([1, 2, 3], SamplingParams(temperature=0.8, max_tokens=2, ignore_eos=True))
    state = torch.random.get_rng_state().clone()
    assert engine.step() == []
    assert torch.equal(state, torch.random.get_rng_state())
    outputs = drain(engine)
    assert len(outputs[0]) == 2
    assert all(0 <= token < model.config.vocab_size for token in outputs[0])


def test_impossible_request_fails_instead_of_spinning(model):
    config = Config(block_size=2, num_blocks=1)
    engine = LLMEngine(config, ModelRunner(model, config))
    engine.add_request([1, 2, 3], SamplingParams())
    with pytest.raises(RuntimeError, match="KV cache capacity"):
        engine.step()
