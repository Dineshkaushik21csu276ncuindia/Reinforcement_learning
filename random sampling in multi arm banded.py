import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.reward_distributions = np.random.normal(0, 1, self.n_arms)
        self.expected_rewards = np.zeros(self.n_arms)
        self.arm_pulls = np.zeros(self.n_arms)

    def pull_arm(self, arm):
        return np.random.normal(self.reward_distributions[arm], 1)

    def update(self, arm, reward):
        self.arm_pulls[arm] += 1
        self.expected_rewards[arm] += (reward -
                                       self.expected_rewards[arm]) / self.arm_pulls[arm]

    def ucb(self, t):
        ucb_values = self.expected_rewards + \
            np.sqrt(2 * np.log(t + 1) / (self.arm_pulls + 1e-6))
        return np.argmax(ucb_values)


def compare_bandit_algorithms(n_arms, n_steps):
    bandit = Bandit(n_arms)
    ucb_rewards = []
    random_rewards = []

    for t in range(n_steps):
        # UCB Algorithm
        ucb_arm = bandit.ucb(t)
        ucb_reward = bandit.pull_arm(ucb_arm)
        bandit.update(ucb_arm, ucb_reward)
        ucb_rewards.append(ucb_reward)

        # Random Sampling
        random_arm = np.random.randint(n_arms)
        random_reward = bandit.pull_arm(random_arm)
        bandit.update(random_arm, random_reward)
        random_rewards.append(random_reward)

    return ucb_rewards, random_rewards


# Simulation parameters
n_arms = 5
n_steps = 1000

ucb_rewards, random_rewards = compare_bandit_algorithms(n_arms, n_steps)

# Plotting cumulative rewards
plt.plot(np.cumsum(ucb_rewards), label='UCB Algorithm')
plt.plot(np.cumsum(random_rewards), label='Random Sampling')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.title('Comparison of UCB Algorithm and Random Sampling in Multi-Armed Bandit')
plt.show()
