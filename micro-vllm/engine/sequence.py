from copy import copy
from enum import Enum, auto
from itertools import count

class Status(Enum):
    WAITING = auto()    # prefill queue
    RUNNING = auto()    # decode queue
    FINISHED = auto()   # finished sequence (eos or max_tokens reached)

class Sequence:
    next_id = count();
    def __init__(self, token_ids: list[int]):
        self.seq_id = next(Sequence.next_id)  
        self.status = Status.WAITING
        self.token_ids = copy(token_ids)
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_scheduled_tokens = 0
        self.num_cached_tokens =  0
        self.block_table = []

        self.is_prefill = True
        self.last_token = token_ids[-1]
        
        self.block_size = sampling_params.block_size
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, i):
        start = i * self.block_size 
        return self.token_ids[start : (start + self.block_size) % self.max_tokens]

