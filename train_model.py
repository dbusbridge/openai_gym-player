import gym

ENVIRONMENT = 'SpaceInvaders-v0'

# Set the environment
env = gym.make(id=ENVIRONMENT)

# Start the environment
env.reset()

# Total reward
total_reward = 0

for i in range(2000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if reward != 0.0:
        total_reward += reward
        print("Gained {reward} points, total score: {total_reward}".format(
            reward=reward,
            total_reward=total_reward))
    if done:
        env.reset()
