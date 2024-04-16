from pyMaze import maze, agent, COLOR, textLabel


def DFS(m, start= None):
    start = (m.rows,m.cols)
    explored = [start]
    frontier = [start]
    dfsPath = {}
    dSearch = []
    while len(frontier)>0:
        currCell = frontier.pop()
        dSearch.append(currCell)
        if currCell == m._goal:
            break
        poss = 0
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d=='E':
                    childCell = (currCell[0], currCell[1]+1)
                elif d=='W':
                    childCell = (currCell[0], currCell[1]-1)        
                elif d=='S':
                    childCell = (currCell[0]+1, currCell[1])
                elif d=='N':
                    childCell = (currCell[0]-1, currCell[1])
                if childCell in explored:
                    continue
                explored.append(childCell)
                frontier.append(childCell )
                dfsPath[childCell] = currCell
        if poss> 1:
            m.markCells.append(currCell)
    fwdPath = {}
    cell = m.__goal
    while cell!= start:
        fwdPath[dfsPath[cell]] = cell
        cell = dfsPath[cell]

    return dSearch,fwdPath, fwdPath

    
if __name__ == '__main__':
    m = maze(5,5)
    m.CreateMaze(loopPercent=100)
    path = DFS(m)

    # agent 
    a = agent(m, footprints=True)
    m.tracePath({a:path})

    #print(m.maze_map)

    m.run()