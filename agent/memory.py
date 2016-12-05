import collections
import random
import config


class Memory(config.AgentConfig):
    def __init__(self):
        self.memory = collections.deque()
        self.count = 0

    def store(self, s_t, a_t, r_t, s_t1, terminal):
        self.memory.append((s_t, a_t, r_t, s_t1, terminal))
        if self.count + 1 > self.replay_memory:
            self.memory.popleft()
        else:
            self.count += 1

    def mini_batch(self):
        mini_batch_sample = random.sample(
            population=self.memory, k=self.batch_size)
        # get the batch variables
        s_j_b = [d[0] for d in mini_batch_sample]
        a_b = [d[1] for d in mini_batch_sample]
        r_b = [d[2] for d in mini_batch_sample]
        s_j1_b = [d[3] for d in mini_batch_sample]
        terminal_b = [d[4] for d in mini_batch_sample]

        return s_j_b, a_b, r_b, s_j1_b, terminal_b
