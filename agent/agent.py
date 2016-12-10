import numpy as np
import config
import random
import tensorflow as tf
import agent.networks as nw
import agent.history as ah
import agent.memory as am


class Agent(config.AgentConfig):
    def __init__(self, game, sess):
        # The game
        self.game = game
        self.sess = sess

        # Learning configuration
        self.epsilon = self.initial_epsilon

        # Take in the binary screen as x_t
        self.x_t = self.game.screen_binary()

        # Make a history of screens, the size depends on config
        self.history = ah.History(x_t_init=self.x_t)

        # Create a memory for replay
        self.memory = am.Memory()

        # Other variables to initialise
        self.update_count = 0
        self.ep_reward = 0.
        self.num_game, self.update_count, self.ep_reward = 0, 0, 0.
        self.total_reward, self.total_loss, self.total_q = 0., 0., 0.
        self.step = 0

        # Arrays
        self.actions = []
        self.ep_rewards = []

        # Neural network
        # Shape
        self.input_layer_shape = (
            [None] + list(self.game.screen_binary().shape) +
            [self.history.history_length])
        self.output_layer_shape = [None] + [self.game.action_space_size]

        # Variables
        with tf.device(self.device):
            self.a = tf.placeholder(
                "float", [None, self.game.action_space_size])
            (self.s, self.q,
             self.q_conv, self.keep_prob) = nw.multilayer_convnet(
                input_layer_shape=self.input_layer_shape,
                output_layer_shape=self.output_layer_shape,
                device=self.device)
        with tf.device(self.device):
            self.q_action = tf.argmax(self.q_conv, dimension=1)
            self.readout_action = tf.reduce_sum(tf.mul(self.q_conv, self.a),
                                                reduction_indices=1)
            self.cost = tf.reduce_mean(tf.square(self.q - self.readout_action))
            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def q_learning_mini_batch(self):
        if self.memory.count < self.history_length:
            return
        else:
            # Pull the mini batch from memory
            s_t_b, a_t_b, r_t_b, s_t1_b, terminal_b = self.memory.mini_batch()

            # Get the estimated values of q for the next time step
            q_t1_b = self.q_conv.eval(
                feed_dict={self.s: s_t1_b,
                           self.keep_prob: self.keep_prob_config})

            # Coerce the terminal into a float binary array
            terminal_b = np.array(terminal_b) + 0.

            # Get the argmax of q for the next time step over all the batch
            max_q_t1_b = np.max(q_t1_b, axis=1)

            # Bellman - calculate the q that we should have estimated
            q_t_b = (1. - terminal_b) * self.gamma * max_q_t1_b + r_t_b

            # Perform gradient step so that next time our estimate of q is
            # better
            self.train_step.run(
                feed_dict={
                    self.q: q_t_b,
                    self.a: [self.hot_one_state(index=action)
                             for action in a_t_b],
                    self.s: s_t_b,
                    self.keep_prob: self.keep_prob_config})

            self.update_count += 1

    def train(self):

        self.sess.run(tf.global_variables_initializer())

        for self.step in range(1, self.max_step):
            if self.step % 100 == 0:
                print(self.step)

            if self.game.do_render:
                self.game.render()

            # Update probability of random action
            if (self.epsilon >
                    self.final_epsilon) and self.step > self.explore_start:
                self.epsilon -= (self.initial_epsilon -
                                 self.final_epsilon) / (self.learn_start -
                                                        self.explore_start)

            # Predict
            a_t = self.predict(self.history.get())
            # Act
            x_t, r_t, terminal, info = self.game.step(a_t)

            # Use the binary representation
            x_bin_t = self.game.screen_binary()

            # Observe
            self.observe(x_t=x_bin_t, r_t=r_t, a_t=a_t, terminal=terminal)

            # If we died
            if terminal:
                print("died")
                # New random game
                x_t, r_t, terminal, info = self.game.new_game()
                # One more game completed
                self.num_game += 1

                # Add the episode reward to the records
                self.ep_rewards.append(self.ep_reward)

                # Reset the episode reward
                self.ep_reward = 0.
            else:
                # Increment the reward for the episode
                self.ep_reward += r_t

            self.actions.append(a_t)
            self.total_reward += r_t

    def predict(self, s_t):
        # Do a random action with probability epsilon
        if random.random() < self.epsilon:
            action = self.game.action_space.sample()
        # Do an action defined by taking the largest estimated Q value
        else:
            action = self.q_action.eval(
                feed_dict={self.s: [s_t],
                           self.keep_prob: self.keep_prob_config})[0]

        return action

    def observe(self, x_t, r_t, a_t, terminal):
        # Get the old history
        s_t = self.history.get()

        # Add the new screen to the history and get the new history
        self.history.add(x_t)
        s_t1 = self.history.get()

        # Add this combination to the memory
        self.memory.store(s_t=s_t, a_t=a_t, r_t=r_t,
                          s_t1=s_t1, terminal=terminal)

        if self.step > self.explore_start:
            self.q_learning_mini_batch()

    def hot_one_state(self, index):
        array = np.zeros(self.game.action_space_size)
        array[index] = 1.
        return array
