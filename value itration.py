import numpy as np

states = [0, 1, 2, 3, 4]
actions = [0, 1]
N_STATES = len(states)
N_ACTIONS = len(actions)

P = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # transition probability
R = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # rewards

P[0, 0, 1] = 1.0
P[1, 1, 2] = 1.0
P[2, 0, 3] = 1.0
P[3, 1, 4] = 1.0
P[4, 0, 4] = 1.0

R[0, 0, 1] = 1
R[1, 1, 2] = 10
R[2, 0, 3] = 100
R[3, 1, 4] = 1000
R[4, 0, 4] = 1.0

gamma = 0.75

# Initialize policy and value arbitrarily
policy = [0 for _ in range(N_STATES)]
V = np.zeros(N_STATES)

print("Initial policy", policy)

# Value Iteration
is_value_changed = True
iterations = 0

while is_value_changed:
    is_value_changed = False
    iterations += 1

    for s in range(N_STATES):
        V[s] = sum([P[s, policy[s], s1] * (R[s, policy[s], s1] +
                   gamma * V[s1]) for s1 in range(N_STATES)])

    for s in range(N_STATES):
        q_best = V[s]
        for a in range(N_ACTIONS):
            q_sa = sum([P[s, a, s1] * (R[s, a, s1] + gamma * V[s1])
                       for s1 in range(N_STATES)])
            if q_sa > q_best:
                policy[s] = a
                q_best = q_sa
                is_value_changed = True

print("Iterations", iterations)
print("Final Policy")
print(policy)
print(V)
