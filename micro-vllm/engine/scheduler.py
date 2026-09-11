from collections import deque 
from sequence import Sequence

class Scheduler: 
    def __init__(self):
        self.waiting : deque[Sequence] = deque()    # waiting to be sched (prefill, or preempted decodes)
        self.running : deque[Sequence] = deque()    # currently running (usually decodes)
    
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq : Sequence):
        self.waiting.append(seq)

    """
    schedules and creates a batch of either waiting or running sequences
    """
    def schedule(self):
        pass

    """
    preempt() removes from the running (decode) queue to free up space for more prefills
    """
    def preempt(self):
        pass

    """
    postprocess() is in charge of handling sequences that have fully finished (reached eos)
    """
    def postprocess(self):
        pass

