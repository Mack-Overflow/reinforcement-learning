import numpy as np

def value_iteration(env, discount_factor=0.9, theta=1e-5):
    # Initialize value function with zeros
    value_func = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = value_func[s]
            # one expectimax of each state
            value_func[s] = max([sum([p * (r + discount_factor * value_func[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.action_space.n)])
            delta = max(delta, abs(v - value_func[s]))
        if delta < theta:
            break
    return value_func

# Use the example environment "FrozenLake-v0" from OpenAI Gym
import gym
env = gym.make("FrozenLake-v1")
value_func = value_iteration(env)
print(value_func)