from pyamaze import maze,COLOR,agent

# Size of the maze
numOfLines = 10
numOfColumns = 10
pop_size = 500
start = (numOfLines, numOfColumns)
goal = (1,1)

m=maze(numOfLines, numOfColumns)
m.CreateMaze(loopPercent=100)
print("map", m.maze_map)

print(m)
a=agent(m,footprints=True,filled=True)

m.tracePath({a:m.path},delay=200,kill=True)
print(m.path)
m.run()