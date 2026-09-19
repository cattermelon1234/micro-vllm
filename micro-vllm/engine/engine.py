"""LLMEngine: the outer loop.

    schedule -> run the model -> postprocess

Owns the scheduler (which owns the block manager) and the model runner.
Everything it does is glue; all the decisions live one layer down.
"""
from dataclasses import dataclass

from scheduler import Scheduler
from sequence import Sequence


@dataclass
class Config:
    # scheduling
    max_seq: int = 8                      # max sequences in one batch
    max_tokens_per_batch: int = 4096      # token budget per forward pass
    # kv cache
    block_size: int = 256                 # tokens per block
    num_blocks: int = 128                 # set by ModelRunner once it measures free memory
    # model
    eos: int = -1                         # filled in from the tokenizer


class LLMEngine:

    def __init__(self, config: Config, model_runner, tokenizer=None):
        # block_size lives in two places and MUST agree, or block_table
        # indices mean different things to Sequence and BlockManager
        Sequence.block_size = config.block_size

        self.config = config
        self.model_runner = model_runner      # .run(seqs, is_prefill) -> list[int]
        self.tokenizer = tokenizer
        self.scheduler = Scheduler(config)

    def add_request(self, prompt, sampling_params):
        """prompt: list[int], or str if a tokenizer was supplied."""
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("String prompts require a tokenizer")
            prompt = self.tokenizer.encode(prompt)
                self.scheduler.add(Sequence(prompt, sampling_params))

    def is_finished(self):
        return self.scheduler.is_finished()

    def step(self):
        """One forward pass. Returns (seq_id, completion_token_ids) for finished seqs."""
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return []

        # the model runner flattens seqs into tensors, runs the model, and
        # samples exactly one token per sequence
        token_ids = self.model_runner.run(seqs, is_prefill)

        self.scheduler.postprocess(seqs, token_ids)
        return [(s.seq_id, s.token_ids[s.num_prompt_tokens:])
                for s in seqs if s.is_finished()]

    def generate(self, prompts, sampling_params):
        for prompt in prompts:
            self.add_request(prompt, sampling_params)

        outputs = {}
        while not self.is_finished():
            for seq_id, token_ids in self.step():
                outputs[seq_id] = token_ids

        ordered = [outputs[seq_id] for seq_id in sorted(outputs)]
        if self.tokenizer is None:
            return ordered
        return [self.tokenizer.decode(t) for t in ordered]
