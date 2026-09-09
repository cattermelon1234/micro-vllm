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

    def reset (self):
        self.token_ids = [] 
        self.hash = -1
        self.ref_count = 1

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

            if token_ids != self.block_list[block_id].token_ids:
                break;

            num_cached += 1

            block = self.block_list[block_id]
            if block.ref_count == 0:
                # means block is unused, but exists in prefix cache. resurrect 
                num_free_blocks += 1
            
        num_free_blocks += seq.num_blocks() - num_cached

        if (len(self.free_blocks) < num_free_blocks):
            return -1

        return num_cached

    def alloc_block(self):
        block_id = self.free_blocks.popleft()
        block = self.block_list[block_id]
        if block.hash != -1 and self.prefix_cache.get(block.hash) == block_id:
            del self.prefix_cache[block.hash]
        block.reset()
        return block_id

    def allocate(self, seq : Sequence):
        num_cached = self.can_allocate(seq)
        h = -1

        for i in range(num_cached):
            token_ids = seq.block(i)
            h = hash((h, tuple(token_ids)))
            block_id = self.prefix_cache[h]
            block = self.block_list[block_id]

            if block.ref_count == 0:
                # resurrect!
                self.free_blocks.remove(block)

            block.ref_count += 1

            seq.block_table.append(self.prefix_cache[h])

        for i in range(num_cached, seq.num_blocks()):
            seq.block_table.append(self.alloc_block())

        seq.num_cached_tokens = num_cached * self.block_size
        return

    def deallocate(self, seq: Sequence):
        # iterate in reverse order to not break hash invariant (think why!)
        for block_id in reversed(seq.block_table):
            block = self.block_list[block_id]
            block.ref_count -= 1

            if block.ref_count == 0:
                self.free_blocks.append(seq.block_table[block_id])

        seq.num_cached_tokens = 0
        seq.block_table = []




