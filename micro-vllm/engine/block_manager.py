from collections import deque
from sequence import Sequence

class Block:
    def __init__ (self, block_id : int):
        self.token_ids = []
        self.hash = -1
        self.block_id = block_id 
        self.ref_count = 0

    def assign (self, prev_hash, token_ids : list[int]):
        self.token_ids = token_ids
        self.hash = hash((prev_hash, tuple(token_ids)))
        self.ref_count += 1

    def reset (self):
        self.token_ids = [] 
        self.hash = -1
        self.ref_count -= 1

class BlockManager:
    def __init__ (self, block_size : int, num_blocks : int):
        self.block_list : list[Block] = [Block(i) for i in range(num_blocks)]
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks : deque[int] = deque(range(self.num_blocks))
        self.prefix_map = {}

    def can_allocate(self, seq : Sequence):
        pass

    def allocate(self, seq : Sequence):
        pass

    def deallocate(self, seq: Sequence):
        pass




