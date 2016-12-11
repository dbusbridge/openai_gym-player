import importlib as il
import tensorflow as tf
import agent.agent as agent
import game.game as game
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sess = tf.InteractiveSession()

il.reload(game)
il.reload(agent)

# Start the game
g = game.Game()

# Start the agent
a = agent.Agent(sess=sess, game=g)

a.train()

import collections
import numpy as np

if len(a.ep_rewards) is not 0:
    # Logging
    a.max_ep_reward = np.max(a.ep_rewards)
    a.min_ep_reward = np.min(a.ep_rewards)
    a.avg_ep_reward = np.mean(a.ep_rewards)

if a.update_count is not 0:
    a.avg_loss = a.total_loss / a.update_count
    a.avg_q = a.total_q / a.update_count

if a.step is not 0:
    a.avg_reward = a.total_reward / a.step

a.epochs_completed = a.step / a.epoch_size

if a.step < a.explore_start:
    a.phase = 'observing'
elif a.step < a.learn_start:
    a.phase = 'exploring'
elif a.step < a.learn_start:
    a.phase = 'training'

status_dict = collections.OrderedDict([
    ('step', a.step),
    ('epoch', a.epochs_completed),
    ('phase', a.phase),
    ('lives', a.game.lives()),
    ('max(ep rew)', a.max_ep_reward),
    ('min(ep rew)', a.min_ep_reward),
    ('avg(ep rew)', a.avg_ep_reward),
    ('avg(loss)', a.avg_loss),
    ('avg(q)', a.avg_q)
])

import pandas as pd


