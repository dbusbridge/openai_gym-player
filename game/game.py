import gym
import config
import numpy as np
import game.screen as screen


class Game(config.GameConfig):
    def __init__(self):
        # Setup the environment according to config
        self.env_name = config.GameConfig.environment
        self.env = gym.make(id=self.env_name)

        # Observation space
        self.observation_space_shape = self.env.observation_space.shape

        # Action space
        self.action_meanings = self.env.get_action_meanings()
        self.action_space = self.env.action_space
        self.action_space_size = self.action_space.n

        # Set the screen
        self.s = screen.Screen(
            screen=np.zeros(shape=self.observation_space_shape))

        # Initialise the game
        _, self.reward, self.terminal, self.info = self.new_game()

    def new_game(self):
        if self.lives() == 0:
            self.s.screen = self.reset()
        (self.s.screen, self.reward,
         self.terminal, self.info) = self.env.step(self.action_space.sample())
        return self.training_screen(), self.reward, self.terminal, self.info

    def reset(self):
        return self.env.reset()

    def step(self, action):
        self.s.screen, self.reward, self.terminal, self.info = self.env.step(
            action)
        return self.training_screen(), self.reward, self.terminal, self.info

    def training_screen(self):
        if self.resize_image:
            return self.s.to_binary().resize().screen
        else:
            return self.s.to_binary().screen

    def lives(self):
        return self.env.ale.lives()
