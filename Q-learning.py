import numpy as np

# Define the environment - a simple grid world
grid_world = [
    ['S', '-', '-', '-', '-'],
    ['-', 'x', '-', '-', '-'],
    ['-', '-', 'x', '-', '-'],
    ['-', '-', '-', 'x', '-'],
    ['-', '-', '-', '-', 'G']
]

# Constants
NUM_ACTIONS = 4  # Up, Down, Left, Right
MAX_EPISODES = 1000
MAX_STEPS = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1

# Function to find available actions in the grid world


def get_available_actions(state):
    row, col = state
    actions = []
    if row > 0 and grid_world[row - 1][col] != 'x':
        actions.append(0)  # Up
    if row < len(grid_world) - 1 and grid_world[row + 1][col] != 'x':
        actions.append(1)  # Down
    if col > 0 and grid_world[row][col - 1] != 'x':
        actions.append(2)  # Left
    if col < len(grid_world[0]) - 1 and grid_world[row][col + 1] != 'x':
        actions.append(3)  # Right
    return actions


# Initialize Q-values for each state-action pair
Q_values = np.zeros((len(grid_world), len(grid_world[0]), NUM_ACTIONS))

# Q-learning algorithm
for episode in range(MAX_EPISODES):
    current_state = (0, 0)  # Start state
    for step in range(MAX_STEPS):
        # Choose action based on epsilon-greedy policy
        if np.random.rand() < EPSILON:
            available_actions = get_available_actions(current_state)
            action = np.random.choice(available_actions)
        else:
            action = np.argmax(Q_values[current_state])

        # Take action and observe next state and reward
        if action == 0:  # Up
            next_state = (current_state[0] - 1, current_state[1])
        elif action == 1:  # Down
            next_state = (current_state[0] + 1, current_state[1])
        elif action == 2:  # Left
            next_state = (current_state[0], current_state[1] - 1)
        else:  # Right
            next_state = (current_state[0], current_state[1] + 1)

        if grid_world[next_state[0]][next_state[1]] == 'G':
            reward = 1  # Goal state
        else:
            reward = 0

        # Update Q-value for the current state-action pair
        Q_values[current_state][action] += LEARNING_RATE * (
            reward + DISCOUNT_FACTOR *
            np.max(Q_values[next_state]) - Q_values[current_state][action]
        )

        current_state = next_state

        if grid_world[current_state[0]][current_state[1]] == 'G':
            break

# Output the learned Q-values
print("Learned Q-values:")
print(Q_values)
