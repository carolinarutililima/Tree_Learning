import gym
import random
import numpy as np
from maze_q_learning import Maze
from pyamaze import maze,agent,COLOR
import matplotlib.pyplot as plt
import tkinter as tk



numOfLines = 5
numOfColumns = 5
m=maze(numOfLines, numOfColumns)
m.CreateMaze(loadMaze= 'maze_5x5.csv' ,theme=COLOR.light)
a = agent(m, footprints=True,color='green',shape='arrow')

maze_map = m.maze_map


env = Maze(maze_map, numOfLines, numOfColumns)

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1 # alpha
discount_rate = 0.9 # gamma
lamb = 0.1 # 0.5, and 0,9

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate =  0.0001

rewards_all_episodes = []

eligibility = np.zeros(shape=(10, 10, 4)) # change here the dimentions to 201 or 301
q_table = np.zeros(shape=(10, 10, 4)) # change here the dimentions to 201 or 301



state = env.reset()
print('State', state)

path = []
best_path = []

for episode in range(num_episodes):

	state = env.reset()
	done = False 
	reward_current_episode = 0

	for step in range(max_steps_per_episode):
		exploration_rate_treseshold = random.uniform(0,1)
		if exploration_rate_treseshold > exploration_rate:

			action = np.argmax(q_table[state[0], state[1], :])
			print('action', action)
		else:
			action = random.randint(0, 3)
			print('action RANDOM', action)

		new_state, reward, done, info = env.step(action)

		path.append(new_state)

		eligibility[state[0],state[1], action] = eligibility[state[0],state[1],action] + 1


		td_error = reward + discount_rate * np.max(q_table[new_state[0], new_state[1], action]) - q_table[state[0], state[1], action]


		q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + learning_rate  * eligibility[state[0], state[1], action] * td_error
	
		eligibility *= lamb * discount_rate

	
		reward_current_episode += reward


		if done:
			if new_state == (1,1): 
				best_path.append(path)
			path =[]
			break
		else:
			state = new_state

	#exploration rate decay 

	exploration_rate =  min_exploration_rate + \
	(max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

	rewards_all_episodes.append(reward_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000


print(best_path)

m.tracePath({a:min(best_path)},delay=200,kill=True)
m.run()