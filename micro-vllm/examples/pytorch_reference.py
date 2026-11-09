"""Tiny random Qwen + paged KV demo; no checkpoint download required.

From micro-vllm:
    PYTHONPATH=../qwen:. ../qwen/.venv/bin/python examples/pytorch_reference.py
"""
import torch

from config import QwenConfig
from qwen import Qwen

from engine.engine import Config, LLMEngine
from engine.model_runner import ModelRunner
from engine.sampling_params import SamplingParams


def main():
    torch.manual_seed(17)
    model = Qwen(QwenConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        tie_word_embeddings=False, bos_token_id=0, eos_token_id=31, pad_token_id=0,
    ))
    config = Config(block_size=4, num_blocks=16, max_tokens_per_batch=3, eos=31)
    runner = ModelRunner(model, config)
    engine = LLMEngine(config, runner)
    prompts = [[1, 2, 3, 4, 5], [6, 7]]
    outputs = engine.generate(prompts, SamplingParams(max_tokens=4, ignore_eos=True))
    for prompt, output in zip(prompts, outputs):
        print(f"{prompt} -> {output}")
    print(f"KV shape: {tuple(runner.kv_cache.k.shape)} (each of K and V)")


if __name__ == "__main__":
    main()
