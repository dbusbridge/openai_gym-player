import numpy as np
import config


class History(config.AgentConfig):
    def __init__(self, x_t_init):
        """
        The History class contains methods for creating and updating the
        history. This is the last n screens component.

        :param dict config: Dictionary of configuration parameters.
        :param np.array x_t_init: The first screen.
        """
        # Create s_t_init, the initial s_t with 4 repeats of the start screen
        # taken from x_t_init
        self.s_t_init = np.stack(
            arrays=(x_t_init, ) * self.history_length,
            axis=2)

        # Set s_t = s_t_init. These are separate objects in case we need to
        # reset later
        self.s_t = self.s_t_init

    def add(self, observation):
        self.s_t[:, :, :-1] = self.s_t[:, :, 1:]
        self.s_t[:, :, -1] = observation
        # self.s_t = np.append(
        #     np.reshape(observation,
        #                newshape=list(observation.shape) + [1]),
        #     self.s_t[:, :, :3], axis=2)

    def reset(self):
        self.s_t = self.s_t_init

    def get(self):
        return self.s_t
