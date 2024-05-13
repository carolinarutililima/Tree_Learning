import numpy as np
from pyamaze import maze,agent,COLOR
import sys
import random

class MyAlgorithm:
    def __init__(self, maze, n_agents, colorList, start=None):
        self.maze = maze
        self.numOfAgents = n_agents
        self.colorList = colorList
        self.start = start if start is not None else (maze.rows, maze.cols)
        self.Q = np.zeros((maze.rows, maze.cols, n_agents,  4))  # Initialize Q-table with dimensions for each cell and action
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.actions = ['N', 'E', 'S', 'W']  # Possible actions
        self.compass = ['N', 'E', 'S', 'W']  # Defining compass to be used in getChildrenPoints


    def choose_action(self, state, agent_index):
        if np.random.rand() < self.epsilon:
            # Explore: Randomly choose an action
            return np.random.choice(self.actions)
        else:
            # Exploit: Choose the best action based on the current Q-table
            # Adjust state indices for 0-based indexing
            state_index = (state[0] - 1, state[1] - 1)
            # Select action with the highest Q-value for the given agent and state
            return self.actions[np.argmax(self.Q[state_index[0], state_index[1], agent_index])]

        

    def update_Q(self, state, action, reward, next_state, agent_index):
        # Adjust indices for zero-based indexing
        state_index = (state[0] - 1, state[1] - 1)
        next_state_index = (next_state[0] - 1, next_state[1] - 1)
        action_index = self.actions.index(action)

        current_Q = self.Q[state_index[0], state_index[1], agent_index, action_index]
        max_future_Q = np.max(self.Q[next_state_index[0], next_state_index[1], agent_index])

        # Additional debugging and assertions to pinpoint the error
        print(f"Current Q: {current_Q}, Max Future Q: {max_future_Q}, Reward: {reward}")

        future_value = self.gamma * max_future_Q
        assert np.isscalar(future_value), "Future value must be a scalar"

        learning_term = self.alpha * (reward + future_value)
        assert np.isscalar(learning_term), "Learning term must be a scalar"

        new_Q = (1 - self.alpha) * current_Q + learning_term
        assert np.isscalar(new_Q), "new_Q must be a scalar"

        self.Q[state_index[0], state_index[1], agent_index, action_index] = new_Q



    # Additional methods for exploring the maze and updating Q-values accordingly...

# When running your simulation:
# - Use `choose_action` to decide the next move based on the current Q-table.
# - After moving, calculate the reward and update the Q-table using `update_Q`.
    # Run the algorithm
    def run(self):

        division = 1.0 / self.numOfAgents
        paths = []
        explored = []
        agents_search = []
        pionner_steps = sys.maxsize
        totalSteps = 0
        for i in range(0, self.numOfAgents):
            start = i * division
            end = (i + 1) * division
            agentInterval = (start, end)
            agentColor = self.colorList[i % len(self.colorList)]

            # Run the algorithm for each agent
            mySearch, effective_path, explored_cells, foundTheGoal = self.run_single_agent(agentInterval, i)
            self.concatenate_new_elements(explored, explored_cells)

            a = agent(self.maze,footprints=True,color=agentColor,shape='square',filled=True)

            paths.append({a:mySearch})
            agents_search.append(mySearch)

            # Number of steps of the agent. Subtract 1 to consider that the first cell is not countable
            agent_steps = len(mySearch) - 1

            # Count the total number of steps
            totalSteps += agent_steps

            # Get the number of the steps of the pionner
            if foundTheGoal == True:
                pionner_steps = agent_steps if pionner_steps > agent_steps else pionner_steps

            self.maze.tracePath(paths[i], kill=False)

            self.maze.run()


        # Get the explored fraction of the maze
        fraction = len(explored) / (self.maze.rows * self.maze.cols)

        # Calculate the fraction of the maze explored until the pionner find the goal
        cells = []
        for i in range(self.numOfAgents):
            aux = agents_search[i][0:pionner_steps]

            for e in aux:
                if e not in cells:
                    cells.append(e)
        fraction_pionner = len(cells) / (self.maze.rows * self.maze.cols)


        return totalSteps, pionner_steps, fraction, fraction_pionner


    def run_single_agent(self, agentInterval, agentIndex):
        currCell = self.start
        foundTheGoal = False
        mySearch = [currCell]  # Start with the initial position
        effective_path = [currCell]
        explored = [currCell]
        parentList = []
        parentList.append((-1,-1))
        
        while not foundTheGoal:
            
            action = self.choose_action(currCell, agentIndex)
            nextCell = self.get_next_state(currCell, action)
            reward, is_terminal = self.calculate_reward(currCell, nextCell)

            # Update Q-values
            self.update_Q(currCell, action, reward, nextCell, agentIndex)

            # Stop if hit a wall (nextCell didn't change)
            if nextCell == currCell:
                print(f"Hit a wall at {nextCell} with action {action}")
                break

            # Update the current cell and paths
            currCell = nextCell
            mySearch.append(currCell)
            effective_path.append(currCell)
            explored.append(currCell)

            if is_terminal:
                foundTheGoal = True

       # return mySearch, effective_path, explored, foundTheGoal


                # Check if there are no non-visited children
            nonVisitedChildren, allChildren = self.getChildrenPoints(currCell, self.maze.maze_map[currCell], parentList[-1], explored)
            if not nonVisitedChildren:
                if currCell not in explored:
                    explored.append(currCell)

                if currCell == self.start:
                    break

                currCell = parentList.pop()
                effective_path.pop()
                continue

            parentList.append(currCell)

            if currCell not in explored:
                explored.append(currCell)

        return mySearch, effective_path, explored, foundTheGoal

    def calculate_reward(self, current, next_state):
        if current == next_state:  # No movement occurred, hit a wall
            return -1, False  # Negative reward for hitting a wall, not terminal
        if next_state == (self.maze.rows, self.maze.cols):  # Check if it's the goal
            return 10, True  # Positive reward for reaching the goal, terminal
        return 0, False  # Standard move with no immediate reward

    

    def get_next_state(self, current, action):
        x, y = current
        next_state = current  # Default to current if no valid move is possible

        if action == 'N' and x > 1:
            next_state = (x - 1, y)
        elif action == 'E' and y < self.maze.cols:
            next_state = (x, y + 1)
        elif action == 'S' and x < self.maze.rows:
            next_state = (x + 1, y)
        elif action == 'W' and y > 1:
            next_state = (x, y - 1)

        # Final boundary check to catch any remaining issues
        if 1 <= next_state[0] <= self.maze.rows and 1 <= next_state[1] <= self.maze.cols:
            return next_state
        else:
            return current  # If calculated next state is still out of bounds, revert to current






    # Return children's cardinal points in preferential order 
    def getChildrenPoints(self, cellCoordinate, cellPoints, parent, explored):
        allChildren = []
        nonVisitedChildren = []
        for d in self.compass:
            if cellPoints[d] == True:
                if d=='N':
                    childCell = (cellCoordinate[0]-1,cellCoordinate[1])
                    if parent == childCell:
                        continue
                    if childCell in explored:
                        allChildren.append('N')
                        continue

                    allChildren.append('N')
                    nonVisitedChildren.append('N')
                elif d=='E':
                    childCell = (cellCoordinate[0],cellCoordinate[1]+1)
                    if parent == childCell:
                        continue
                    if childCell in explored:
                        allChildren.append('E')
                        continue

                    allChildren.append('E')
                    nonVisitedChildren.append('E')
                elif d=='S':
                    childCell = (cellCoordinate[0]+1,cellCoordinate[1])
                    if parent == childCell:
                        continue
                    if childCell in explored:
                        allChildren.append('S')
                        continue

                    allChildren.append('S')
                    nonVisitedChildren.append('S')
                elif d=='W':
                    childCell = (cellCoordinate[0],cellCoordinate[1]-1)
                    if parent == childCell:
                        continue
                    if childCell in explored:
                        allChildren.append('W')
                        continue

                    allChildren.append('W')
                    nonVisitedChildren.append('W')

        return nonVisitedChildren, allChildren

    # If there is child to visit, this function will decide what child to visit
    # This function doesn't work if there is no child to visit
    def defineAgentNextStep(self, agentInterval, agent_path, allChildren, nonVisitedChildren, currCell, agentIndex):

        totalNumberOfChildren = len(allChildren)
        
        # If there is only 1 child just go on
        if totalNumberOfChildren == 1:
            agent_path.append((-1, -1))
            return 0
        
        # Get the weight interval of each child
        relative_node_weights = self.getRelativeNodeWeights(agent_path, totalNumberOfChildren)


        if self.filledInterval[agentIndex] == False:
            # Return the first child that is able to obey the limits
            for i in range(0, totalNumberOfChildren):

                # If the current child's interval is on the right of the agent's interval, surely the agent finished its interval
                if agentInterval[1] <= relative_node_weights[i][0]:
                    self.filledInterval[agentIndex] = True
                    break

                nodeIsInsideAgentInterval = agentInterval[0] < relative_node_weights[i][1] and agentInterval[1] > relative_node_weights[i][0]
                nodeWasNotVisistedByTheAgent = allChildren[i] in nonVisitedChildren

                if nodeIsInsideAgentInterval and nodeWasNotVisistedByTheAgent:
                    agent_path.append((i, totalNumberOfChildren))
                    return i

            # If the agent is in the root and it doesn't find a node that fills the requirement, it finished its interval
            # It occurs when the agent come back to root but it has already been visited the root's children
            # In the context of this Bachelor's thesis, the agent will stop
            if currCell == self.start:
                self.filledInterval[agentIndex] = True

            # If the agent doesn't finish its interval and no node that fills the requirement was found, it goes to parent
            if self.filledInterval[agentIndex] == False:
                return -1
                    
        # The agent surely finished its interval, and it will do a dummy DFS
        for i in range(0, totalNumberOfChildren):
            if allChildren[i] in nonVisitedChildren:
                agent_path.append((i, totalNumberOfChildren))
                return i

    def getRelativeNodeWeights(self, agent_path, count_children):

        # Calculating the previous node interval according the agent path
        # It is the way that we can calculate values related to a number less than 1 in a mixed radix
        path_size = len(agent_path)
        node_interval = (0, 1)
        if path_size > 0:
            if agent_path[0][0] != -1: # only if the first related node has more than one child
                chunk = 1 / agent_path[0][1]
                node_interval = (agent_path[0][0] * chunk,  agent_path[0][0] * chunk + chunk)

            for i in range(1, path_size):
                if agent_path[i][0] == -1:
                    continue

                node_interval_size = node_interval[1] - node_interval[0]
                chunk = node_interval_size / agent_path[i][1]
                node_interval = (node_interval[0] + agent_path[i][0] * chunk, node_interval[0] + agent_path[i][0] * chunk + chunk)

        # Calculating the weights of the next nodes
        weights = []
        node_interval_size = node_interval[1] - node_interval[0]
        chunk = node_interval_size / count_children
        start = 0
        end = 0
        for i in range(0, count_children):
            start = node_interval[0] + chunk * i
            end = start + chunk
            weight = (start, end)
            weights.append(weight)

        return weights

    # Auxiliary function to print agent color
    def getColorString(self, color):
        if color == COLOR.red:
            return "Vermelho"
        elif color == COLOR.blue:
            return "Azul"
        elif color == COLOR.yellow:
            return "Amarelo"
        elif color == COLOR.cyan:
            return "Ciano"
        elif color == COLOR.black:
            return "Preto"
        elif color == COLOR.pink:
            return "Rosa"
        elif color == COLOR.orange:
            return "Laranja"

    # Auxiliary function to print agent interval
    def getIntervalString(self, interval):
        if interval[1] > 0.9999:
            return "[" + str(interval[0]) + ", 1]"
        else:
            return "[" + str(interval[0]) + ", " + str(interval[1]) + "["
        
    # Auxiliary function to print the next direction
    def next_direction(self, current, next):
        if current[0] != next[0]:
            if current[0] < next[0]:
                return "S"
            else:
                return "N"
        else:
            if current[1] < next[1]:
                return "E"
            else:
                return "W"

    # Auxiliary function to get the mixed radix representation of the agent effective_path
    def getMixedRadixRepresentation(self, effective_path, maze):
        mixedRadix = [(0, 0)]
        
        for i in range(0, len(effective_path) - 1):
            radix = 0
            next = self.next_direction(effective_path[i], effective_path[i + 1])

            directions = []
            if maze.maze_map[effective_path[i]]['N'] == 1 and (effective_path[i][0] - 1, effective_path[i][1]) != effective_path[i - 1]:
                radix += 1
                directions.append('N')
            if maze.maze_map[effective_path[i]]['E'] == 1 and (effective_path[i][0], effective_path[i][1] + 1) != effective_path[i - 1]:
                radix += 1
                directions.append('E')
            if maze.maze_map[effective_path[i]]['S'] == 1 and (effective_path[i][0] + 1, effective_path[i][1]) != effective_path[i - 1]:
                radix += 1
                directions.append('S')
            if maze.maze_map[effective_path[i]]['W'] == 1 and (effective_path[i][0], effective_path[i][1] - 1) != effective_path[i - 1]:
                radix += 1
                directions.append('W')

            digit = directions.index(next)

            if radix == 1:
                radix = "X"
                digit = "X"

            mixedRadix.append((digit, radix))

        return mixedRadix
    
    # Auxiliary function to concatenate to add only new elements to array
    def concatenate_new_elements(self, main, vector):
        for e in vector:
            if e not in main:
                main.append(e)
    