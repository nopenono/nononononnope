import gym
import random
env = gym.make('FrozenLake-v0')
numGames = 1000
numSteps = 10
env.reset()
print(env.observation_space)
print(env.action_space)
score = 0

# 0 = Left
# 1 = Down
# 2 = Right
# 3 = Up
def PlayGame():
    global score
    for i in range (numSteps):
        obs, rew, done, info = env.step(0)
        env.render()
        if done:
            score += rew
            break

for g in range(numGames):
    env.reset()
    PlayGame()

print(score)