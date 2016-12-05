import game
import importlib as il
import agent.agent as agent
import tensorflow as tf

sess = tf.InteractiveSession()

# Start the game
g = game.Game()

# Start the agent
a = agent.Agent(sess=sess, game=g)

# Total reward
total_reward = 0

a.train()

a.q_learning_mini_batch()


for i in range(20000):
    g.render()
    action = g.action_space.sample() # your agent here (this takes random actions)
    g.step(action)
    if g.reward != 0.0:
        total_reward += g .reward
        print("Gained {reward} points, total score: {total_reward}".format(
            reward=g.reward,
            total_reward=total_reward))
    if g.terminal:
        g.reset()
