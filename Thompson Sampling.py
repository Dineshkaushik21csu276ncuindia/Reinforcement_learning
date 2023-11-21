import numpy as np


def UCB_algorithm(k, T):
    Q = np.zeros(k)
    N = np.zeros(k)
    for t in range(T):
        UCB = Q + np.sqrt(np.log(t + 1) / N)
        arm = np.argmax(UCB)
        reward = np.random.binomial(1, 0.5)
        Q[arm] = (Q[arm] * N[arm] + reward) / (N[arm] + 1)
        N[arm] += 1
    return Q


def random_sampling(k, T):
    Q = np.zeros(k)
    for t in range(T):
        arm = np.random.randint(0, k)
        reward = np.random.binomial(1, 0.5)
        Q[arm] = (Q[arm] * t + reward) / (t + 1)
    return Q


k = 10  # Number of arms
T = 1000  # Number of timesteps

UCB_reward = UCB_algorithm(k, T)
random_reward = random_sampling(k, T)

print("UCB reward:", UCB_reward)
print("Random reward:", random_reward)
