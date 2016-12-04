import numpy as np
import config
import agent.history as ah
import agent.memory as am


class Agent(config.AgentConfig):
    def __init__(self, game, sess):
        self.gamma = config.AgentConfig.gamma
        self.sess = sess
        self.game = game

        # Take in the binary screen as x_t
        self.x_t = self.game.screen_binary()

        # Make a history of screens, the size depends on config
        self.history = ah.History(x_t_init=self.x_t)

        # Create a memory for replay
        self.memory = am.Memory()

    def q_learning_mini_batch(self):
        if self.memory.count < config.AgentConfig.history_length:
            return
        else:
            # Pull the mini batch from memory
            s_t_b, a_t_b, r_t_b, s_t1_b, terminal_b = self.memory.mini_batch()

            # Get the estimated values of q for the next time step
            q_t1_b = self.target_q.eval({self.target_s_t: s_t1_b})

            # Coerce the terminal into a float binary array
            terminal_b = np.array(terminal_b) + 0.

            # Get the argmax of q for the next time step over all the batch
            max_q_t1_b = np.max(q_t1_b, axis=1)

            # Bellman - calculate the q that we should have estimated
            target_q_t_b = (1. - terminal_b) * self.gamma * max_q_t1_b + r_t_b

            # Perform gradient step so that next time our estimate of q is
            # better
            self.train_step.run(
                feed_dict={
                    self.target_q_t: target_q_t_b,
                    self.action: a_t_b,
                    self.s_t: s_t_b)

            self.update_count += 1











#
#
#
#
#
#
#
#
#
# import os
# import time
# import random
# import numpy as np
# from tqdm import tqdm
# import tensorflow as tf
#
# from .base import BaseModel
# from .history import History
# from .ops import linear, conv2d
# from .replay_memory import ReplayMemory
# from utils import get_time, save_pkl, load_pkl
#
# class Agent(BaseModel):
#   def __init__(self, config, environment, sess):
#     super(Agent, self).__init__(config)
#     self.sess = sess
#     self.weight_dir = 'weights'
#
#     self.env = environment
#     self.history = History(self.config)
#     self.memory = ReplayMemory(self.config, self.model_dir)
#
#     with tf.variable_scope('step'):
#       self.step_op = tf.Variable(0, trainable=False, name='step')
#       self.step_input = tf.placeholder('int32', None, name='step_input')
#       self.step_assign_op = self.step_op.assign(self.step_input)
#
#     self.build_dqn()
#
#   def train(self):
#     start_step = self.step_op.eval()
#     start_time = time.time()
#
#     num_game, self.update_count, ep_reward = 0, 0, 0.
#     total_reward, self.total_loss, self.total_q = 0., 0., 0.
#     max_avg_ep_reward = 0
#     ep_rewards, actions = [], []
#
#     screen, reward, action, terminal = self.env.new_random_game()
#
#     for _ in range(self.history_length):
#       self.history.add(screen)
#
#     for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
#       if self.step == self.learn_start:
#         num_game, self.update_count, ep_reward = 0, 0, 0.
#         total_reward, self.total_loss, self.total_q = 0., 0., 0.
#         ep_rewards, actions = [], []
#
#       # 1. predict
#       action = self.predict(self.history.get())
#       # 2. act
#       screen, reward, terminal = self.env.act(action, is_training=True)
#       # 3. observe
#       self.observe(screen, reward, action, terminal)
#
#       if terminal:
#         screen, reward, action, terminal = self.env.new_random_game()
#
#         num_game += 1
#         ep_rewards.append(ep_reward)
#         ep_reward = 0.
#       else:
#         ep_reward += reward
#
#       actions.append(action)
#       total_reward += reward
#
#       if self.step >= self.learn_start:
#         if self.step % self.test_step == self.test_step - 1:
#           avg_reward = total_reward / self.test_step
#           avg_loss = self.total_loss / self.update_count
#           avg_q = self.total_q / self.update_count
#
#           try:
#             max_ep_reward = np.max(ep_rewards)
#             min_ep_reward = np.min(ep_rewards)
#             avg_ep_reward = np.mean(ep_rewards)
#           except:
#             max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0
#
#           print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
#               % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)
#
#           if max_avg_ep_reward * 0.9 <= avg_ep_reward:
#             self.step_assign_op.eval({self.step_input: self.step + 1})
#             self.save_model(self.step + 1)
#
#             max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)
#
#           if self.step > 180:
#             self.inject_summary({
#                 'average.reward': avg_reward,
#                 'average.loss': avg_loss,
#                 'average.q': avg_q,
#                 'episode.max reward': max_ep_reward,
#                 'episode.min reward': min_ep_reward,
#                 'episode.avg reward': avg_ep_reward,
#                 'episode.num of game': num_game,
#                 'episode.rewards': ep_rewards,
#                 'episode.actions': actions,
#                 'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
#               }, self.step)
#
#           num_game = 0
#           total_reward = 0.
#           self.total_loss = 0.
#           self.total_q = 0.
#           self.update_count = 0
#           ep_reward = 0.
#           ep_rewards = []
#           actions = []
#
#   def predict(self, s_t, test_ep=None):
#     ep = test_ep or (self.ep_end +
#         max(0., (self.ep_start - self.ep_end)
#           * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
#
#     if random.random() < ep:
#       action = random.randrange(self.env.action_size)
#     else:
#       action = self.q_action.eval({self.s_t: [s_t]})[0]
#
#     return action
#
#   def observe(self, screen, reward, action, terminal):
#     reward = max(self.min_reward, min(self.max_reward, reward))
#
#     self.history.add(screen)
#     self.memory.add(screen, reward, action, terminal)
#
#     if self.step > self.learn_start:
#       if self.step % self.train_frequency == 0:
#         self.q_learning_mini_batch()
#
#       if self.step % self.target_q_update_step == self.target_q_update_step - 1:
#         self.update_target_q_network()
#
#   def q_learning_mini_batch(self):
#     if self.memory.count < self.history_length:
#       return
#     else:
#       s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()
#
#     t = time.time()
#     if self.double_q:
#       # Double Q-learning
#       pred_action = self.q_action.eval({self.s_t: s_t_plus_1})
#
#       q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
#         self.target_s_t: s_t_plus_1,
#         self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
#       })
#       target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward
#     else:
#       q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
#
#       terminal = np.array(terminal) + 0.
#       max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
#       target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward
#
#     _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
#       self.target_q_t: target_q_t,
#       self.action: action,
#       self.s_t: s_t,
#       self.learning_rate_step: self.step,
#     })
#
#     self.writer.add_summary(summary_str, self.step)
#     self.total_loss += loss
#     self.total_q += q_t.mean()
#     self.update_count += 1
#
#   def build_dqn(self):
#     self.w = {}
#     self.t_w = {}
#
#     #initializer = tf.contrib.layers.xavier_initializer()
#     initializer = tf.truncated_normal_initializer(0, 0.02)
#     activation_fn = tf.nn.relu
#
#     # training network
#     with tf.variable_scope('prediction'):
#       if self.cnn_format == 'NHWC':
#         self.s_t = tf.placeholder('float32',
#             [None, self.screen_height, self.screen_width, self.history_length], name='s_t')
#       else:
#         self.s_t = tf.placeholder('float32',
#             [None, self.history_length, self.screen_height, self.screen_width], name='s_t')
#
#       self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
#           32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
#       self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
#           64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
#       self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
#           64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')
#
#       shape = self.l3.get_shape().as_list()
#       self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
#
#       if self.dueling:
#         self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
#             linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')
#
#         self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
#             linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')
#
#         self.value, self.w['val_w_out'], self.w['val_w_b'] = \
#           linear(self.value_hid, 1, name='value_out')
#
#         self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
#           linear(self.adv_hid, self.env.action_size, name='adv_out')
#
#         # Average Dueling
#         self.q = self.value + (self.advantage -
#           tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
#       else:
#         self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
#         self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')
#
#       self.q_action = tf.argmax(self.q, dimension=1)
#
#       q_summary = []
#       avg_q = tf.reduce_mean(self.q, 0)
#       for idx in xrange(self.env.action_size):
#         q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
#       self.q_summary = tf.merge_summary(q_summary, 'q_summary')
#
#     # target network
#
#
#     with tf.variable_scope('summary'):
#       scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
#           'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']
#
#       self.summary_placeholders = {}
#       self.summary_ops = {}
#
#       for tag in scalar_summary_tags:
#         self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
#         self.summary_ops[tag]  = tf.scalar_summary("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])
#
#       histogram_summary_tags = ['episode.rewards', 'episode.actions']
#
#       for tag in histogram_summary_tags:
#         self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
#         self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])
#
#       self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)
#
#     tf.initialize_all_variables().run()
#
#     self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)
#
#     self.load_model()
#     self.update_target_q_network()
#
#   def update_target_q_network(self):
#     for name in self.w.keys():
#       self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})
#
#   def save_weight_to_pkl(self):
#     if not os.path.exists(self.weight_dir):
#       os.makedirs(self.weight_dir)
#
#     for name in self.w.keys():
#       save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))
#
#   def load_weight_from_pkl(self, cpu_mode=False):
#     with tf.variable_scope('load_pred_from_pkl'):
#       self.w_input = {}
#       self.w_assign_op = {}
#
#       for name in self.w.keys():
#         self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
#         self.w_assign_op[name] = self.w[name].assign(self.w_input[name])
#
#     for name in self.w.keys():
#       self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})
#
#     self.update_target_q_network()
#
#   def inject_summary(self, tag_dict, step):
#     summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
#       self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
#     })
#     for summary_str in summary_str_lists:
#       self.writer.add_summary(summary_str, self.step)
#
#   def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
#     if test_ep == None:
#       test_ep = self.ep_end
#
#     test_history = History(self.config)
#
#     if not self.display:
#       gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
#       self.env.env.monitor.start(gym_dir)
#
#     best_reward, best_idx = 0, 0
#     for idx in xrange(n_episode):
#       screen, reward, action, terminal = self.env.new_random_game()
#       current_reward = 0
#
#       for _ in range(self.history_length):
#         test_history.add(screen)
#
#       for t in tqdm(range(n_step), ncols=70):
#         # 1. predict
#         action = self.predict(test_history.get(), test_ep)
#         # 2. act
#         screen, reward, terminal = self.env.act(action, is_training=False)
#         # 3. observe
#         test_history.add(screen)
#
#         current_reward += reward
#         if terminal:
#           break
#
#       if current_reward > best_reward:
#         best_reward = current_reward
#         best_idx = idx
#
#       print "="*30
#       print " [%d] Best reward : %d" % (best_idx, best_reward)
#       print "="*30
#
#     if not self.display:
#       self.env.env.monitor.close()
#       #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')
