import config
import network.networks.deepmnist_convnet as deepmnist_convnet
import network.networks.deepmind_convnet as deepmind_convnet


class NetworkSwitch(config.AgentConfig):
    def __init__(self, input_layer_shape, output_layer_shape,
                 device='/cpu:0'):
        if self.network_choice == 'DeepMind':
            self.network = deepmind_convnet.deepmind_convnet(
                input_layer_shape=input_layer_shape,
                output_layer_shape=output_layer_shape,
                device=device)
        elif self.network_choice == 'DeepMNIST':
            self.network = deepmnist_convnet.deepmnist_convnet(
                input_layer_shape,
                output_layer_shape,
                device=device)
