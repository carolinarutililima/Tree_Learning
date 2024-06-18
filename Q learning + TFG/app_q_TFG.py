import numpy as np
from pyamaze import maze,agent,COLOR
from env_q_TFG_Q_update_back_go import MyAlgorithm


# List of colors of the agents
colorList = [COLOR.red, COLOR.blue, COLOR.yellow, COLOR.cyan, COLOR.black]

numOfLines = 8
numOfColumns = 8
n_agents = 2
m=maze(numOfLines, numOfColumns)
#m.CreateMaze(theme=COLOR.light, saveMaze='maze_8x8')

#  maze_5x5  maze_6x6 maze_7x7 maze_8x8 maze_8x8_v2 maze_10x10

m.CreateMaze(loadMaze= 'maze_8x8_v2.csv', theme=COLOR.light)
maze_map = m.maze_map


myAlgorithm = MyAlgorithm(m, n_agents,  maze_map, colorList, (numOfLines, numOfColumns))
steps, pionner_steps, fraction, fraction_pionner = myAlgorithm.run() 
