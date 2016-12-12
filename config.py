class AgentConfig(object):
    device = '/gpu:0'
    history_length = 4
    replay_memory = 1000000
    batch_size = 32
    gamma = 0.99
    explore_start = 50000
    learn_start = 100000
    max_step = 2000000
    initial_epsilon = 1
    final_epsilon = 0.001
    keep_prob_config = 1
    network_choice = 'DeepMind'
    # network_choice = 'DeepMNIST'
    restore_from_saved_model = True
    model_dir = 'saved_models/Breakout-v0'
    model_restore_steps = 200000
    steps_save = 100000
    epoch_size = 250000


class GameConfig(object):
    environment = 'Breakout-v0'
    do_render = False
    resize_image = True
    resize_shape = (84, 84)
