import importlib as il
import tensorflow as tf
import agent.agent as agent
import game.game as game

sess = tf.InteractiveSession()

il.reload(game)
il.reload(agent)

# Start the game
g = game.Game()

# Start the agent
a = agent.Agent(sess=sess, game=g)

a.train()

g.lives()
