"""Readable PyTorch reference runner for the sibling Qwen implementation.

Runs sequences separately using one shared, fixed KV pool. No packed attention,
Triton, CUDA graphs, tensor parallelism, or automatic memory sizing yet.
"""
import torch

from .kv_cache import PagedKVCache, SequenceCache


class ModelRunner:
    def __init__(self, model, config):
        self.model = model.eval()
        parameter = next(model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.kv_cache = PagedKVCache(
            model.config,
            num_blocks=config.num_blocks,
            block_size=config.block_size,
            device=self.device,
            dtype=self.dtype,
        )

    @torch.inference_mode()
    def forward(self, seq):
        """Logits for exactly the scheduled tokens: [new_tokens, vocabulary]."""
        start = seq.num_cached_tokens
        end = start + seq.num_scheduled_tokens
        if start < 0 or not start < end <= seq.num_tokens:
            raise ValueError("Scheduled tokens must be a nonempty uncached sequence slice")
        if end > self.model.config.max_position_embeddings:
            raise ValueError("Scheduled tokens exceed the model context limit")
        required_blocks = (end + self.kv_cache.block_size - 1) // self.kv_cache.block_size
        if len(seq.block_table) < required_blocks:
            raise ValueError("Sequence has too few allocated KV blocks")
        if any(b < 0 or b >= self.kv_cache.num_blocks for b in seq.block_table):
            raise ValueError("Sequence contains an invalid physical KV block ID")

        input_ids = torch.tensor(seq.token_ids[start:end], device=self.device, dtype=torch.long)
        positions = torch.arange(start, end, device=self.device)
        block_table = torch.tensor(seq.block_table, device=self.device, dtype=torch.long)
        cache = SequenceCache(self.kv_cache, block_table, positions)

        # Absolute positions matter for both RoPE and causal masking. For a
        # chunk starting at 8, its first query may read keys 0..8, not just key 0.
        key_positions = torch.arange(end, device=self.device)
        allowed = key_positions[None, :] <= positions[:, None]
        mask = torch.zeros((end - start, end), device=self.device, dtype=self.dtype)
        mask.masked_fill_(~allowed, float("-inf"))

        return self.model(
            input_ids=input_ids.unsqueeze(0),
            position_ids=positions.unsqueeze(0),
            attention_mask=mask[None, None, :, :],
            cache=cache,
        )[0]

    @torch.inference_mode()
    def run(self, seqs, is_prefill):
        """Engine interface: one token ID per scheduled sequence, in order."""
        if not is_prefill and any(seq.num_scheduled_tokens != 1 for seq in seqs):
            raise ValueError("Decode requires one scheduled token per sequence")

        token_ids = []
        for seq in seqs:
            logits = self.forward(seq)[-1].float()
            if seq.is_prefill:
                # Still inside a prompt chunk. The scheduler discards this
                # placeholder; don't consume randomness sampling unused logits.
                token_id = 0
            elif seq.temperature == 0:
                token_id = logits.argmax().item()
            else:
                probs = torch.softmax(logits / seq.temperature, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
            token_ids.append(token_id)
        return token_ids
