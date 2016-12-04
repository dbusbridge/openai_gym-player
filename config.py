class AgentConfig(object):
    history_length = 4
    replay_memory = 10
    batch_size = 5
    gamma = 0.99


class GameConfig(object):
    environment = 'SpaceInvaders-v0'
    do_render = False
