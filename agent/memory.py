import collections
import random
import config


class Memory(config.AgentConfig):
    def __init__(self):
        self.memory = collections.deque()
        self.replay_memory = config.AgentConfig.replay_memory
        self.batch_size = config.AgentConfig.batch_size
        self.count = 0

    def store(self, s_t, a_t, r_t, s_t1, done):
        self.memory.append((s_t, a_t, r_t, s_t1, done))
        if self.count + 1 > self.replay_memory:
            self.memory.popleft()
        else:
            self.count += 1

    def mini_batch(self):
        mini_batch_sample = random.sample(
            population=self.memory, k=self.batch_size)
        # get the batch variables
        s_j_batch = [d[0] for d in mini_batch_sample]
        a_batch = [d[1] for d in mini_batch_sample]
        r_batch = [d[2] for d in mini_batch_sample]
        s_j1_batch = [d[3] for d in mini_batch_sample]

        return s_j_batch, a_batch, r_batch, s_j1_batch
