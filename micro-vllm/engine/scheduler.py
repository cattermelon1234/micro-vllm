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
        prefill = True

        # schedule a prefill batch
        if self.waiting:
            while (len(batch) < self.max_seq and num_tokens < self.max_tokens_per_batch):
                seq = self.waiting[0]
                while self.running and self.block_manager.can_allocate(seq) == -1:
                    victim = self.running.pop()
                    self.preempt(victim)

                cached_tokens = self.block_manager.allocate(seq)

                if cached_tokens == -1:
                    break

                # calculate num scheduled tokens
                remaining = seq.num_tokens - seq.num_cached_tokens
                budget = self.max_tokens_per_batch - num_tokens

                num_scheduled = min(remaining, budget)
                seq.num_scheduled_tokens = num_scheduled

                num_tokens += seq.num_scheduled_tokens
                batch.append(seq)

                # only append to running if not chunked prefill, if chunked prefill, keep in waiting
                if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                    seq.status = Status.RUNNING 
                    self.waiting.popleft() 
                    self.running.append(seq)

        # construct a decode batch
        else:
            prefill = False
            while (len(batch) < self.max_seq and num_tokens < self.max_tokens_per_batch):
                seq = self.running.popleft()
                if self.block_manager.can_append(seq):
                    self.block_manager.try_append(seq)
                    self.running.popleft()
                    num_tokens += seq.num_scheduled_tokens
                    batch.append(seq)
                    self.running.append(seq)

                    seq.num_scheduled_tokens = 1
                else:
                    # preempt running seq if not enough space for a decode
                    self.preempt(seq)
                    self.running.append(seq)
                    continue

        return batch, prefill
            

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
    def postprocess(self, seqs : list[Sequence], token_ids: list[int], is_prefill : bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.publish_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # check for chunked prefill, stop once reaching end
            # do so before appending, because this result will be garbage for chunked prefill
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            seq.append_token(token_id)

            if (not seq.ignore_eos and token_id == self.eos) or \
                       seq.num_completion_tokens() == seq.max_tokens:
                        seq.status = Status.FINISHED
                        self.block_manager.deallocate(seq)
                        self.running.remove(seq)


