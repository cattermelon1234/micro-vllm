from collections import deque 
from sequence import Sequence, Status
from block_manager import BlockManager

class Scheduler: 
    def __init__(self, config : Config):
        self.waiting : deque[Sequence] = deque()    # waiting to be sched (prefill, or preempted decodes)
        self.running : deque[Sequence] = deque()    # currently running (usually decodes)

        self.max_seq = config.max_seq 
        self.max_tokens_per_batch = config.max_tokens_per_batch;
        self.eos = config.eos
        self.block_manager = BlockManager(config.block_size, config.num_blocks)
    
    def is_finished(self):
        return (not self.waiting and not self.running)

    def add(self, seq : Sequence):
        self.waiting.append(seq)

    """
    schedules and creates a batch of either waiting or running sequences
    """
    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.is_finished():
            return ([], False)

        num_tokens = 0
        batch : list[Sequence] = []

        # prefill
        # runs first, so new requests take priority over decoding ones
        while self.waiting and len(batch) < self.max_seq:
            seq = self.waiting[0]

            budget = self.max_tokens_per_batch - num_tokens
            if budget == 0:
                break

            # only alloc blocks if block table doesnt exist to avoid duplication
            if not seq.block_table:
                num_cached = self.block_manager.can_allocate(seq)
                if num_cached == -1:
                    # no more room, stop. we could evict, but that leads to issues
                    # for prefill evictions
                    break
                remaining = seq.num_tokens - num_cached * self.block_manager.block_size
            else:
                remaining = seq.num_tokens - seq.num_cached_tokens

            # only chunk a prompt that cannot fit in a whole empty batch.
            # anything else waits one step and goes through in a single pass,
            # instead of being split across two for no benefit
            if budget < remaining and batch:
                break

            # alloc blocks only once every break above is cleared, so we never
            # reserve memory for a sequence we then decline to schedule
            if not seq.block_table:
                self.block_manager.allocate(seq)

            seq.num_scheduled_tokens = min(remaining, budget)
            num_tokens += seq.num_scheduled_tokens
            batch.append(seq)

            # migrate to the decode queue only once the whole prompt is covered;
            # a chunked prefill stays in self.waiting for its next pass
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = Status.RUNNING
                seq.is_prefill = False
                self.waiting.popleft()
                self.running.append(seq)

        if batch:
            return batch, True

        # decode
        # only reached when prefill scheduled nothing, so the engine still makes
        # progress while the waiting queue is blocked on memory
        while self.running and len(batch) < self.max_seq:
            seq = self.running.popleft()

            # make room for this sequence's next block, evicting the newest
            # running sequences first. already-batched sequences are out of
            # self.running right now, so they can never be chosen as victims
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)       # nothing left to evict but myself
                    seq = None
                    break

            if seq is None:
                break

            self.block_manager.try_append(seq)
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            num_tokens += seq.num_scheduled_tokens
            batch.append(seq)

        # put the scheduled sequences back at the front in preserved order
        self.running.extendleft(reversed(batch))

        return batch, False


    """
    preempt() removes from the running (decode) queue to free up space for more prefills
    """
    def preempt(self, seq : Sequence):
        seq.status = Status.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    """
    postprocess() runs after every step of LLM Engine (each run batch)
    token_ids is a list of generated tokens, 1 per seq scheduled
    """
    def postprocess(self, seqs : list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.publish_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # a sequence still mid-prefill sampled its token from a mid-prompt
            # logit, which predicts a token already in the prompt. discard it.
            # schedule() clears is_prefill on the pass that finishes the prompt,
            # so this is per-sequence rather than per-batch
            if seq.is_prefill:
                continue

            seq.append_token(token_id)

            if (not seq.ignore_eos and token_id == self.eos) or \
                       seq.num_completion_tokens() == seq.max_tokens:
                        seq.status = Status.FINISHED
                        self.block_manager.deallocate(seq)
                        self.running.remove(seq)


