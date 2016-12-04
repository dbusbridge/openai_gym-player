import cv2
import gym
from matplotlib import pyplot as plt


class Game:
    def __init__(self, config):
        # Setup the environment according to config
        self.env_name = config['environment']
        self.env = gym.make(id=self.env_name)

        # Observation space
        self.observation_space_shape = self.env.observation_space.shape

        # Action space
        self.action_meanings = self.env.get_action_meanings()
        self.action_space = self.env.action_space
        self.action_space_size = self.action_space.n

        # Initialise the game
        self.screen = self.reset()

        # Initialise other variables
        self.reward = 0
        self.done = False
        self.info = None

        # Rendering
        self.do_render = config['do_render']
        if self.do_render:
            cv2.startWindowThread()
            cv2.namedWindow(winname=self.env_name)

    def render_grid(self):
        plt.subplot(231), plt.imshow(
            self.screen, 'gray'), plt.title('RGB')

        plt.subplot(232), plt.imshow(
            self.screen_bgr(), 'gray'), plt.title('BGR')

        plt.subplot(233), plt.imshow(
            self.screen_grey(), 'gray'), plt.title('Grey')

        plt.subplot(234), plt.imshow(
            self.screen_binary(), 'gray'), plt.title('Binary')

    def reset(self):
        return self.env.reset()

    def screen_bgr(self):
        return cv2.cvtColor(self.screen, cv2.COLOR_RGB2BGR)

    def screen_grey(self):
        return cv2.cvtColor(self.screen, cv2.COLOR_BGR2GRAY)

    def screen_binary(self):
        result, screen_binary = cv2.threshold(
            src=self.screen_grey(),
            thresh=1, maxval=255,
            type=cv2.THRESH_BINARY)
        return screen_binary

    def step(self, action):
        self.screen, self.reward, self.done, self.info = self.env.step(action)
        return self

    def render(self):
        cv2.imshow(winname=self.env_name,
                   mat=self.screen_bgr())
