import gym
import random
import numpy as np
from maze_q_learning import Maze
from pyamaze import maze, agent, COLOR
import matplotlib.pyplot as plt
import tkinter as tk

n_agents = 1

numOfLines = 5
numOfColumns = 5
m = maze(numOfLines, numOfColumns)
m.CreateMaze(loadMaze='maze_5x5.csv', theme=COLOR.light)
agents = []

maze_map = m.maze_map
env = Maze(maze_map, numOfLines, numOfColumns)

state_shape = (2,)  # Assuming each state vector has 2 elements (x, y coordinates)
state = np.zeros((n_agents,) + state_shape, dtype=int)  # Create a 2D array to store states

colorList = [COLOR.red, COLOR.blue, COLOR.yellow, COLOR.cyan, COLOR.black]
shapeList = ['arrow', 'square', 'arrow', 'square']

for i in range(n_agents):
    character = agent(m, footprints=True, color=colorList[i], shape=shapeList[i])
    agents.append(character)

num_episodes = 1000
max_steps_per_episode = 100

learning_rate = 0.1  # alpha
discount_rate = 0.9  # gamma
lamb = 0.1  # 0.5, and 0.9

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.0001

rewards_all_episodes = []

eligibility = np.zeros(shape=(10, 10, 4))  # Single eligibility trace for all agents
q_table = np.zeros(shape=(10, 10, 4))  # Single Q-table for all agents
path = [[] for _ in range(n_agents)]  # Create a list of empty lists for each agent
best_path = [[] for _ in range(n_agents)]

end = [[] for _ in range(n_agents)]  # Create a list of empty lists for each agent

for episode in range(num_episodes):
    for i in range(n_agents):
        state[i] = env.reset()
        print(state[i])

    path = [[] for _ in range(n_agents)]  # Create a list of empty lists for each agent

    done = np.zeros(n_agents, dtype=bool)
    reward_current_episode = np.zeros(n_agents)

    for step in range(max_steps_per_episode):
        for i in range(n_agents):
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state[i][0], state[i][1], :])
                print('action', action)
            else:
                action = random.randint(0, 3)
                print('action RANDOM', action)

            new_state, reward, done_agent, info = env.step(action, state[i][0], state[i][1])
            done[i] = done_agent

            path[i].append(new_state)

            # Increment the eligibility trace for the current state-action pair
            eligibility[state[i][0], state[i][1], action] += 1

            # Calculate the TD error using the correct maximum Q-value from the new state
            td_error = reward + discount_rate * np.max(q_table[new_state[0], new_state[1], :]) - q_table[state[i][0], state[i][1], action]

            # Update the Q-value for the current state-action pair
            q_table[state[i][0], state[i][1], action] += learning_rate * eligibility[state[i][0], state[i][1], action] * td_error

            # Apply the decay to the eligibility trace for all state-action pairs
            eligibility *= lamb * discount_rate

            reward_current_episode[i] += reward

            if done[i]:
                if new_state == (1, 1):
                    best_path[i].append(path[i])
                    end[i].append(True)
                    print("Agent", i)
                    print("FOUND THE GOAL")
                state[i] = (numOfLines, numOfColumns)
                path[i] = [(numOfLines, numOfColumns)]
                break
            else:
                state[i] = new_state
                print("state i", state[i])

    # Exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(reward_current_episode)


# Calculate the number of steps each agent took to reach the goal
steps_to_goal = [len(min(best_path[i])) for i in range(n_agents)]

for i in range(n_agents):
    print(f"Agent {i+1} took {steps_to_goal[i]} steps to reach the goal.")

dict_agents = {}

for i in range(n_agents):
    dict_agents[agents[i]] = min(best_path[i])


m.tracePath(dict_agents, delay=200, kill=True)
m.run()


