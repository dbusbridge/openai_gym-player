import game
import importlib as il
import agent.agent as agent
import tensorflow as tf

sess = tf.InteractiveSession()

il.reload(game)
il.reload(agent)

# Start the game
g = game.Game()

# Start the agent
a = agent.Agent(sess=sess, game=g)

a.train()
