class AgentConfig(object):
    history_length = 4
    replay_memory = 10
    batch_size = 5
    gamma = 0.99
    explore_start = 1000
    learn_start = 10000
    max_step = 20000
    initial_epsilon = 1
    final_epsilon = 0.001
    keep_prob_config = 0.95


class GameConfig(object):
    environment = 'SpaceInvaders-v0'
    do_render = False
