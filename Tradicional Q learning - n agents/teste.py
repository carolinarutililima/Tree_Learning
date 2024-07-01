import gym
import random
import numpy as np
from maze_q_learning import Maze
from pyamaze import maze,agent,COLOR
import matplotlib.pyplot as plt
import tkinter as tk


n_agents = 3

numOfLines = 5
numOfColumns = 5
m=maze(numOfLines, numOfColumns)
m.CreateMaze(loadMaze= 'maze_5x5.csv' ,theme=COLOR.light)
agents = []
a = agent(m, footprints=True,color='green',shape='arrow')


state_shape = (2,)  # Assuming each state vector has 4 elements
state = np.zeros((n_agents,) + state_shape, dtype=int)  # Create a 2D array to store states



colorList = [COLOR.red, COLOR.blue, COLOR.yellow, COLOR.cyan, COLOR.black]
shapeList = ['arrow', 'square', 'arrow', 'square' ]

for i in range(0, n_agents):
	character = chr(97 + i)
	character = agent(m, footprints=True,color=colorList[i],shape=shapeList[i])
	agents.append(character)

#best_path = [[[(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)]], [[(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)]]]

best_path = [[[(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (1, 4), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (1, 4), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 3), (1, 2), (1, 3), (1, 2), (1, 1)]], [[(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(4, 5), (3, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 3), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 5), (2, 4), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)]], [[(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (2, 4), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(4, 5), (3, 5), (2, 5), (2, 4), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)], [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 5), (1, 5), (1, 4), (1, 5), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (1, 3), (1, 2), (1, 1)]]]

# Calculate the number of steps each agent took to reach the goal
steps_to_goal = [len(min(best_path[i])) for i in range(n_agents)]

print(steps_to_goal)
for i in range(n_agents):
    print(f"Agent {i+1} took {steps_to_goal[i]} steps to reach the goal.")
    

dict_agents = {}

for i in range(0, n_agents):
    dict_agents[agents[i]] = min(best_path[i])


print(dict_agents)



m.tracePath(dict_agents, delay=200,kill=True)
#m.tracePath({a:min(best_path)},delay=200,kill=True)
m.run()

