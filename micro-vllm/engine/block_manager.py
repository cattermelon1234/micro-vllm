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
        self.prefix_cache = {}

    def can_allocate(self, seq : Sequence):
        h = -1
        num_cached = 0
        num_free_blocks = 0
        for i in range(seq.num_blocks() - 1):
            token_ids = seq.block(i)
            h = hash((h, tuple(token_ids)))
            if h not in self.prefix_cache:
                break
            
            block_id = self.prefix_cache[h]
            num_cached += 1

            block = self.block_list[block_id]
            if block.ref_count == 0:
                # means block is unused, but exists in prefix cache. resurrect 
                num_free_blocks += 1

            block.ref_count += 1
            
            num_free_blocks += seq.num_blocks() - num_cached
            return num_cached

    def alloc_block(self, token_ids : list[int]):
        block_id = self.free_blocks.pop()
        del self.prefix_cache[block_id]
        return block_id

    def allocate(self, seq : Sequence):
        num_cached = self.can_allocate(seq)
        h = -1
        if not num_cached:
            return 

        for i in range(num_cached):
            token_ids = seq.block(i)
            h = hash((h, tuple(token_ids)))

            seq.block_table.append(self.prefix_cache[h])

        for i in range(num_cached, seq.num_blocks()):
            seq.block_table.append(self.alloc_block(token_ids))

        return

    def deallocate(self, seq: Sequence):
        # iterate in reverse order to not break hash invariant (think why!)
        for i in range(seq.num_blocks(), -1, -1):
            block = seq.block_table[i]
            block.ref_count -= 1

            if block.ref_count == 0:
                self.free_blocks.append(seq.block_table[i])
        seq.num_cached_tokens = 0
        seq.block_table = []




