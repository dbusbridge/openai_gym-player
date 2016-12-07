# openai_gym-space_invaders

## Introduction

This repo contains code that trains a reinforcement learning algorithm on the [OpenAI Gym](https://gym.openai.com) environment [SpaceInvaders-v0](https://gym.openai.com/envs/SpaceInvaders-v0). It is loosely based on the repos:

+ [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)

+ [Deep Flappy Bird](https://github.com/yenchenlin/DeepLearningFlappyBird)

Code is not yet fully functioning. A partially functioning version of the code can be found on the [basic_framework](https://github.com/dbusbridge/openai_gym-space_invaders/tree/basic_framework) branch.

## Environment details

### Challenge

Maximize your score in the Atari 2600 game SpaceInvaders. In this environment, the observation is an RGB image of the screen, which is an array of shape (210, 160, 3) Each action is repeatedly performed for a duration of `k` frames, where `k` is uniformly sampled from {2, 3, 4}.

### Notes

`SpaceInvaders-v0` is an unsolved environment, which means it does not have a specified reward threshold at which it's considered solved.

The game is simulated through the *Arcade Learning Environment [ALE]*, which uses the *Stella [Stella]* Atari emulator.
