import gym
import random
import numpy as np
from maze_q_learning import Maze
from pyamaze import maze,agent,COLOR
import matplotlib.pyplot as plt
import tkinter as tk


n_agents = 2

numOfLines = 5
numOfColumns = 5
m=maze(numOfLines, numOfColumns)
m.CreateMaze(loadMaze= 'maze_5x5.csv' ,theme=COLOR.light)
agents = []


state_shape = (2,)  # Assuming each state vector has 4 elements
state = np.zeros((n_agents,) + state_shape, dtype=int)  # Create a 2D array to store states



for i in range(0, n_agents):
	character = chr(97 + i)
	character = agent(m, footprints=True,color='green',shape='arrow')
	agents.append(character)

best_path = [[[(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)]], [[(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)]]]


dict_agents = {}

for i in range(0, n_agents):
    dict_agents[agents[i]] = min(best_path[i])


m.tracePath(dict_agents, delay=200,kill=True)
m.run()

