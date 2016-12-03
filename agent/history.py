import numpy as np


class History:
	def __init__(self, config, x_t_init):
		"""
		The History class contains methods for creating and updating the
		history. This is the last n screens component.

		:param dict config: Dictionary of configuration parameters.
		:param np.array x_t_init: The first screen.
		"""

		# Create s_t_init, the initial s_t with 4 repeats of the start screen
		# taken from x_t_init
		self.s_t_init = np.stack(
			arrays=(x_t_init, ) * config.history_length,
			axis=2)

		# Set s_t = s_t_init. These are seperate objects in case we need to
		# reset later
		self.s_t = self.s_t_init

	def add(self, observation):
		self.s_t[:-1] = self.s_t[1:]
		self.s_t[-1] = observation

	def reset(self):
		self.s_t = self.s_t_init

	def get(self):
		return self.s_t
