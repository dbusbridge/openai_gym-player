import agent.history as ah


class Agent:
    def __init__(self, env, sess, config):
        self.config = config
        self.sess = sess
        self.env = env

        self.x_t = self.env.reset()

        self.history = ah.History(config=self.config,
                                  x_t_init=self.x_t)
