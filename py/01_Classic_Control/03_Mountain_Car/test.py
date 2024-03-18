import gymnasium as gym
env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)
print(env.action_space.high)