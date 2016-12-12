# openai_gym-player

## Introduction

This repo contains code that trains a reinforcement learning algorithm on the [OpenAI Gym](https://gym.openai.com) Atari environments (although you can alter the environment in the config file and it *should* all work with environments that are two dimensional in input fine; nothing depends on the environment except the assumption of a two dimensional observation space and a one dimensional control space). It is inspired by the implementations (**not by me!**):

+ [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)

+ [Deep Flappy Bird](https://github.com/yenchenlin/DeepLearningFlappyBird)

## Dependencies

+ An anaconda distribution

+ opencv

## Details

### Network architectures used

#### `DeepMind`

| Layer | Input                | Filter size | Strides | Number of filters | Activation function | Output size       |
|-------|----------------------|-------------|---------|-------------------|---------------------|-------------------|
| conv1 | 84 x 84 x 4          | 8 x 8       | 4       | 32                | ReLU                | 20 x 20 x 32      |
| conv2 | 20 x 20 x 32         | 4 x 4       | 2       | 64                | ReLU                | 9 x 9 x 64        |
| conv3 | 9 x 9 x 64           | 3 x 3       | 1       | 64                | ReLU                | 7 x 7 x 64        |
| fc1   | flattened 7 x 7 x 64 |             |         | 512               | ReLU                | 512               |
| fc2   |                      |             |         | Number of actions | Linear              | Number of actions |

