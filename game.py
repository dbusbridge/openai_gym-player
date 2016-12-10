import cv2
import gym
# from matplotlib import pyplot as plt
import config
import numpy as np


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
        self.s = Screen(screen=np.zeros(shape=self.observation_space_shape))

        # Initialise the game
        _, self.reward, self.terminal, self.info = self.new_game()

        # Rendering
        self.do_render = config.GameConfig.do_render
        if self.do_render:
            cv2.startWindowThread()
            cv2.namedWindow(winname=self.env_name)

    def new_game(self):
        self.s.screen = self.reset()
        self.reward = 0
        self.terminal = False
        self.info = None
        return self.training_screen(), self.reward, self.terminal, self.info
    #
    # def render_grid(self):
    #     plt.subplot(231), plt.imshow(
    #         self.screen, 'gray'), plt.title('RGB')
    #
    #     plt.subplot(232), plt.imshow(
    #         self.screen_bgr(), 'gray'), plt.title('BGR')
    #
    #     plt.subplot(233), plt.imshow(
    #         self.screen_grey(), 'gray'), plt.title('Grey')
    #
    #     plt.subplot(234), plt.imshow(
    #         self.screen_binary(), 'gray'), plt.title('Binary')

    def reset(self):
        return self.env.reset()

    def step(self, action):
        self.s.screen, self.reward, self.terminal, self.info = self.env.step(
            action)
        return self.training_screen(), self.reward, self.terminal, self.info

    def render(self):
        cv2.imshow(winname=self.env_name,
                   mat=self.s.to_rgb().screen)

    def training_screen(self):
        if self.resize_image:
            return self.s.to_binary().resize().screen
        else:
            return self.s.to_binary().screen


class Screen(config.GameConfig):
    def __init__(self, screen):
        self.screen = screen

    def copy(self):
        new_obj = Screen(screen=self.screen)
        return new_obj

    def to_colour(self, cv2COLOR, inplace=False):
        if inplace:
            new_obj = self
        else:
            new_obj = self.copy()
        new_obj.screen = cv2.cvtColor(new_obj.screen, cv2COLOR)
        return new_obj

    def to_rgb(self, inplace=False):
        return self.to_colour(cv2COLOR=cv2.COLOR_BGR2RGB, inplace=inplace)

    def to_grey(self, inplace=False):
        return self.to_colour(cv2COLOR=cv2.COLOR_BGR2GRAY, inplace=inplace)

    def to_binary(self, inplace=False):
        new_obj = self.to_grey(inplace=inplace)
        result, new_obj.screen = cv2.threshold(
            src=new_obj.screen,
            thresh=1, maxval=255,
            type=cv2.THRESH_BINARY)
        new_obj.screen = new_obj.screen / 255.
        return new_obj

    def resize(self, inplace=False):
        if inplace:
            new_obj = self
        else:
            new_obj = self.copy()
        new_obj.screen = cv2.resize(new_obj.screen, new_obj.resize_shape)
        return new_obj
