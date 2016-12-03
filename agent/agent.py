import tensorflow as tf
import agent.history


class Agent:
	def __init__(self, env, sess, device):
		self.sess = sess
		self.env = env
		self.device = device

		self.history = agent.history.History()
