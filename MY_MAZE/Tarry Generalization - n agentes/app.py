import numpy as np
from pyamaze import maze,agent,COLOR
from env import TarryGeneralization


# List of colors of the agents
colorList = [COLOR.red, COLOR.blue, COLOR.yellow, COLOR.cyan, COLOR.black]

numOfLines = 5
numOfColumns = 5
n_agents = 3
m=maze(numOfLines, numOfColumns)
m.CreateMaze(loadMaze= 'maze_5x5.csv' ,theme=COLOR.light)
a = agent(m, footprints=True,color='green',shape='arrow')


tarryGeneralization = TarryGeneralization(m, n_agents,  colorList, (numOfLines, numOfColumns))
steps, pionner_steps, fraction, last_steps = tarryGeneralization.run()
    

