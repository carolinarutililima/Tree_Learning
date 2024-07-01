import numpy as np
from pyamaze import maze,agent,COLOR
from env import MyAlgorithm


# List of colors of the agents
colorList = [COLOR.red, COLOR.blue, COLOR.yellow, COLOR.cyan, COLOR.black]

numOfLines = 9
numOfColumns = 9
n_agents = 5
m=maze(numOfLines, numOfColumns)
m.CreateMaze(loadMaze= 'maze_9x9_v5.csv', theme=COLOR.light)



myAlgorithm = MyAlgorithm(m, n_agents,  colorList, (numOfLines, numOfColumns))
steps, pionner_steps, fraction, fraction_pionner = myAlgorithm.run() 
