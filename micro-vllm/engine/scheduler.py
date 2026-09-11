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
        return not self.waiting and not self.running

    def add(self, seq : Sequence):
        self.waiting.append(seq)

    """
    schedules and creates a batch of either waiting or running sequences
    """
    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.is_finished:
            return ([], False)
        num_tokens = 0
        batch : list[Sequence] = []

        # schedule a prefill batch
        if self.waiting:
            while (len(batch) < self.max_seq and num_tokens < self.max_tokens_per_batch):
                seq = self.waiting[0]
                while not self.block_manager.can_allocate(seq):
                    victim = self.running.popleft()
                    self.preempt(victim)

                self.block_manager.allocate(seq)
                self.waiting.popleft()
                num_tokens += seq.num_tokens
                batch.append(seq)


        # construct a decode batch
        else:
            while (len(batch) < self.max_seq and num_tokens < self.max_tokens_per_batch):
                seq = self.running[0]
                if self.block_manager.can_append(seq):
                    self.block_manager.try_append(seq)
                    self.waiting.popleft()
                    num_tokens += seq.num_tokens
                    batch.append(seq)

        return batch, True
            

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
    """
    def postprocess(self, seqs : list[Sequence], is_prefill : bool):
        for seq in seqs:
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            if (not seq.ignore_eos and seq.last_token == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = Status.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)


