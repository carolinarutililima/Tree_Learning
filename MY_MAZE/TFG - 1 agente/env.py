import numpy as np
from pyamaze import maze,agent,COLOR
import sys
import random

class TarryGeneralization:
    def __init__(self, maze, colorList, start=None):
        self.maze = maze
        self.numOfAgents = 1
        self.colorList = colorList
        self.start = start

        if self.start is None:
            self.start = (self.maze.rows,self.maze.cols)

    # Run the algorithm
    def run(self):

        paths = []
        agents_search, fraction = self.run_agents()
        pionner_steps = sys.maxsize
        last_steps = 0
        totalSteps = 0

        for i in range(0, len(agents_search)):
            agentColor = self.colorList[i % len(self.colorList)]
            a = agent(self.maze,footprints=True,color=agentColor,shape='square',filled=True)
            paths.append({a:agents_search[i]})

            # Number of steps of the agent. Subtract 1 to consider that the first cell is not countable
            agent_steps = len(agents_search[i]) - 1

            # Count the total number of steps
            totalSteps += agent_steps

            # Get the number of the steps of the pionner
            pionner_steps = agent_steps if pionner_steps > agent_steps else pionner_steps

            # Get the number of the steps of the last agent that found the goal
            last_steps = agent_steps if last_steps < agent_steps else last_steps


        # Show only agent i
        # self.maze.tracePaths([paths[2]], kill=False, delay=100)
        print(paths)
        self.maze.tracePath(paths[i], kill=False, delay=50)
        ##self.maze.tracePaths_by_key_press(paths, kill=False)

        self.maze.run()

        return totalSteps, pionner_steps, fraction, last_steps

    # Run the algorithm for all agents
    # The following steps come from the article "Multi-Agent Maze Exploration" - Kivelevitch and Cohen 2010
    def run_agents(self):

        # Matrix of the search of each agent
        agents_search = []

        # Matrix of the effective of each agent
        # It will be trully computed only in the PHASE 1 of the algorithm
        # In PHASE 2 it is useful only to know the effective path of the pionner agent
        effective_path = []

        # Matrix of the explored cells by each agent 
        agents_exploredCells = []

        # Array of the current cell of each agent
        agents_currentCell = []

        # Matrix of the parents list of each agent
        agents_parents = []

        # Array of the explored cells
        exploredCells = []
        exploredCells.append(self.start)

        # Array of dead-end cells
        deadEndCells = []

        # Array of booleans that indicates if an agent found the goal
        foundTheGoal = []

        # Matrix of the Last Common Location (LCL) of each agent related to the others
        lcl = []


        for i in range(0, self.numOfAgents):
            agents_search.append([self.start])
            effective_path.append([self.start])
            agents_exploredCells.append([self.start])
            agents_currentCell.append(self.start)
            agents_parents.append([(-1,-1)])
            foundTheGoal.append(False)
            lcl.append([])

            for j in range(0, self.numOfAgents):
                lcl[i].append(self.start)

        
        # Pay attention: this algorithm is divided in two phases. In the first phase, the agents
        # follow 6 steps until some agent finds the goal. If some agent finds the goal, the second
        # phase will start. In the second phase, the agents that doesn't find the goal, will go to
        # the Last Common Location (LCL) with the pionner agent, and then the agents will follow the
        # goal path

        # PHASE 1
        while True not in foundTheGoal:

            # During each loop all the agents follow the steps below
            # Step 1: The agent should move to cells that have not been traveled by any agent
            # Step 2: If there are several such cells, the agent should choose one arbitrarily
            # Step 3: If there is no cell that has not been traveled by an agent, the agent should prefer to move to a cell that has not been traveled by it
            # Step 4: If all the possible directions have already been traveled by the agent, or if the agent has reached a dead-end, the agent should retreat until a cell that meets one of the previous conditions
            # Step 5: All the steps should be logged by the agent in its history
            # Step 6: When retreating, mark the cells retreated from as “dead end”
            for i in range(0, self.numOfAgents):

                # If the agent found the goal, go to the next agent
                if foundTheGoal[i] == True:
                    continue

                # Get cell children
                currentCell = agents_currentCell[i]
                parent = agents_parents[i][-1]
                children = self.getChildren(currentCell, self.maze.maze_map[currentCell], parent)

                # Check cell children
                visited = []
                nonVisited = []
                for child in children:
                    if child not in deadEndCells:
                        if child in exploredCells:
                            visited.append(child)
                        else:
                            nonVisited.append(child)

                # If there is non visited cells, choose one arbitrarily
                if len(nonVisited) > 0:
                    # Update parents list
                    agents_parents[i].append(agents_currentCell[i])

                    # Go to child
                    agents_currentCell[i] = nonVisited[random.randint(0, len(nonVisited) - 1)]

                    # Update the general array of explored cells
                    if agents_currentCell[i] not in exploredCells:
                        exploredCells.append(agents_currentCell[i])

                    # Update the array of the explored cells by the agent
                    if agents_currentCell[i] not in agents_exploredCells[i]:
                        agents_exploredCells[i].append(agents_currentCell[i])

                    # Update agent's effective path
                    effective_path[i].append(agents_currentCell[i])

                # If there is no cell that has not been visted by an agent, the agent should prefer to move to a cell that has not been visted by it
                elif len(visited) > 0:
                    agent_nonVisited = []

                    for child in visited:
                        if child not in agents_exploredCells[i]:
                            agent_nonVisited.append(child)

                    # Update parents list
                    agents_parents[i].append(agents_currentCell[i])

                    # Go to child
                    agents_currentCell[i] = agent_nonVisited[random.randint(0, len(agent_nonVisited) - 1)]

                    # Update the array of the explored cells by the agent
                    agents_exploredCells[i].append(agents_currentCell[i])

                    # Update agent's effective path
                    effective_path[i].append(agents_currentCell[i])


                # No children - dead-end
                else:
                    deadEndCells.append(agents_currentCell[i])

                    # Go to parent
                    agents_currentCell[i] = agents_parents[i].pop()
                    effective_path[i].pop()

                # Update agent search path
                agents_search[i].append(agents_currentCell[i])

                # Update the Last Common Location (LCL) matrix if it is necessary
                for j in range(0, self.numOfAgents):
                    if i == j:
                        continue

                    if agents_currentCell[i] in effective_path[j]:
                        lcl[i][j] = agents_currentCell[i]
                        lcl[j][i] = agents_currentCell[i]

                # Check if the agent found the goal
                if agents_currentCell[i] == self.maze._goal:
                    foundTheGoal[i] = True
                    break

        # PHASE 2
        # Get the pionner index
        pionner = foundTheGoal.index(True)

        # From the article: "Given the path of the agent that was the 
        # first to find the exit, the pioneer, each agent has to find the last location
        # from its own-logged history that matches a location on the path
        # of the pioneer. This location is denoted Last Common Location (LCL)"
        for i in range(0, self.numOfAgents):
            if i == pionner:
                continue

            # Move agent until the last common location with the pionner agent
            comeback_index = -2
            while agents_search[i][-1] != lcl[pionner][i]:
                # Come back unitl the LCL
                agents_search[i].append(effective_path[i][comeback_index])
                comeback_index -= 1

            # Add the path from the last common location until the goal
            lcl_index = effective_path[pionner].index(lcl[pionner][i])
            split_index = lcl_index + 1
            agents_search[i] += effective_path[pionner][split_index:]

        # Get the explored fraction of the maze
        fraction = len(exploredCells) / (self.maze.rows * self.maze.cols)

        return agents_search, fraction
    
    # Return cell children 
    def getChildren(self, cellCoordinate, cellPoints, parent):
        children = []

        for d in "NESW":
            if cellPoints[d] == True:
                if d=='N':
                    childCell = (cellCoordinate[0]-1,cellCoordinate[1])
                    if parent == childCell:
                        continue

                    children.append(childCell)
                elif d=='E':
                    childCell = (cellCoordinate[0],cellCoordinate[1]+1)
                    if parent == childCell:
                        continue

                    children.append(childCell)
                elif d=='S':
                    childCell = (cellCoordinate[0]+1,cellCoordinate[1])
                    if parent == childCell:
                        continue

                    children.append(childCell)
                elif d=='W':
                    childCell = (cellCoordinate[0],cellCoordinate[1]-1)
                    if parent == childCell:
                        continue

                    children.append(childCell)

        return children