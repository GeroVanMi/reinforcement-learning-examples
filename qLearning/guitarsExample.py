import numpy as np

gamma = 0.75
alpha = 0.9

location_to_state = {
    'L1': 0,
    'L2': 1,
    'L3': 2,
    'L4': 3,
    'L5': 4,
    'L6': 5,
    'L7': 6,
    'L8': 7,
    'L9': 8,
}
state_to_location = dict((state, location) for location, state in location_to_state.items())

actions = list(range(0, 9))

rewards = np.array(
    [[0, 1, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 1, 0, 1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1, 0, 1, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 1, 0]]
)


def q_learning(rewards):
    """
    This is the function that implements the q-Learning algorithm.
    :param rewards:
    :return:
    """
    q_values = np.zeros([9, 9])

    for step in range(1000):
        # Choose a random starting state to explore the environment (this can't be the start location)
        current_state = np.random.randint(0, 9)

        playable_actions = []

        for potential_action in range(9):
            if rewards[current_state, potential_action] > 0:
                playable_actions.append(potential_action)

        next_state = np.random.choice(playable_actions)
        reward = rewards[current_state, next_state]
        expected_future_return = q_values[next_state, np.argmax(q_values[next_state,])]
        sampled_return = reward + gamma * expected_future_return

        expected_return = q_values[current_state, next_state]

        temporal_difference = sampled_return - expected_return
        q_values[current_state, next_state] += alpha * temporal_difference
    return q_values


def get_optimal_route(start_location: str, end_location: str):
    # This is essentially a table of state-action pairs
    rewards_copy = np.copy(rewards)
    terminal_state = location_to_state[end_location]

    # Set high reward for terminal state.
    rewards_copy[terminal_state] = 999

    q_values = q_learning(rewards_copy)

    route = [start_location]
    next_location = start_location
    while next_location != end_location:
        starting_state = location_to_state[next_location]
        next_state = np.argmax(q_values[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)

    return route


route_l9 = get_optimal_route('L9', 'L1')
print(route_l9)
