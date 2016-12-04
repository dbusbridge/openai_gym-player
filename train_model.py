import game as g

# Set the environment
CONFIG = {
    'history_length': 4,
    'environment': 'SpaceInvaders-v0',
    'do_render': True
}

# Start the game
game = g.Game(config=CONFIG)

# Total reward
total_reward = 0

for i in range(2000):
    game.render()
    action = game.action_space.sample() # your agent here (this takes random actions)
    game.step(action)
    if game.reward != 0.0:
        total_reward += game.reward
        print("Gained {reward} points, total score: {total_reward}".format(
            reward=game.reward,
            total_reward=total_reward))
    if game.done:
        game.reset()
