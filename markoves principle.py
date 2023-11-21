import numpy as np

# Define transition probabilities for a simple 3-state Markov chain
transition_matrix = np.array([
    [0.1, 0.7, 0.2],  # Transition probabilities from state 0 to states 0, 1, 2
    [0.4, 0.1, 0.5],  # Transition probabilities from state 1 to states 0, 1, 2
    [0.6, 0.3, 0.1]   # Transition probabilities from state 2 to states 0, 1, 2
])

# Initial state probabilities
# Starting probabilities for states 0, 1, 2
initial_state_probs = np.array([0.3, 0.4, 0.3])

# Simulate the Markov chain
num_steps = 10
current_state = np.random.choice(
    range(len(initial_state_probs)), p=initial_state_probs)

print("Simulated Markov Chain:")
print(f"Step 0: State {current_state}")

for step in range(1, num_steps + 1):
    current_state = np.random.choice(
        range(len(transition_matrix[current_state])), p=transition_matrix[current_state])
    print(f"Step {step}: State {current_state}")
