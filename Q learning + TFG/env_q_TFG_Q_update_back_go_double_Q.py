import numpy as np
from pyamaze import maze,agent,COLOR
import sys
import random
import copy

class MyAlgorithm:
    def __init__(self, maze, n_agents, maze_map, colorList, start=None):
        self.maze = maze
        self.maze_map = maze_map
        self.numOfAgents = n_agents
        self.colorList = colorList
        self.start = start if start is not None else (maze.rows, maze.cols)
        self.Q = np.zeros((maze.rows, maze.cols, n_agents,  4))  # Initialize Q-table with dimensions for each cell and action
        self.learning_rate = 0.1  # Learning rate
        self.discount_rate = 0.9  # Discount factor
        self.lamb = 0.1  # Exploration rate
        self.actions = ['N', 'E', 'S', 'W']  # Possible actions
        self.filledInterval = [False for i in range(n_agents)]
        self.agent_intervals = [(i / n_agents, (i + 1) / n_agents) for i in range(n_agents)]
        self._agent_done =[False for i in range(n_agents)]
        self.Q_hit = np.zeros((maze.rows, maze.cols,  4))  # Initialize Q-table with dimensions for each cell and action


    def choose_action(self, state, agent_index, exploration_rate):

        exploration_rate_treseshold = random.uniform(0,1)
        if exploration_rate_treseshold > exploration_rate:
            # Exploit: Choose the best action based on the current Q-table
            # Adjust state indices for 0-based indexing
            state_index = (state[0] - 1, state[1] - 1)
            # Select action with the highest Q-value for the given agent and state
            return self.actions[np.argmax(self.Q[state_index[0], state_index[1], agent_index])]
        else:
            # Explore: Randomly choose an action
            return np.random.choice(self.actions)


        

    def update_Q(self, state, action, reward, next_state, agent_index):
        # Adjust indices for zero-based indexing
        state_index = (state[0] - 1, state[1] - 1)
        next_state_index = (next_state[0] - 1, next_state[1] - 1)
        action_index = self.actions.index(action)

        current_Q = self.Q_hit[state_index[0], state_index[1], action_index]
        max_future_Q = np.max(self.Q_hit[next_state_index[0], next_state_index[1]])

        # Additional debugging and assertions to pinpoint the error
        print(f"Current Q: {current_Q}, Max Future Q: {max_future_Q}, Reward: {reward}")

        future_value = self.discount_rate * max_future_Q
        assert np.isscalar(future_value), "Future value must be a scalar"

        learning_term = self.learning_rate * (reward + future_value - current_Q)
        assert np.isscalar(learning_term), "Learning term must be a scalar"

        new_Q = current_Q + learning_term
        assert np.isscalar(new_Q), "new_Q must be a scalar"

        self.Q_hit[state_index[0], state_index[1], action_index] = new_Q

        self.Q[:, :, agent_index, :] = self.Q_hit

    def update_Q_before(self, state, explored, agentIndex, action, next_state, Another_agent):
        
        count = explored.count(state)

        # Adjust indices for zero-based indexing
        state_index = (state[0] - 1, state[1] - 1)
        next_state_index = (next_state[0] - 1, next_state[1] - 1)
        action_index = self.actions.index(action)

        current_Q = self.Q[state_index[0], state_index[1], agentIndex, action_index]
        max_future_Q = np.max(self.Q[next_state_index[0], next_state_index[1], agentIndex])

        if count > 1: 
            weight = -0.5
           
        elif Another_agent  == True:
            weight = 0.1

        else: 
            weight = -0.1


        reward = (len(explored) * weight)

        # Additional debugging and assertions to pinpoint the error
        print(f"Current Q: {current_Q}, Max Future Q: {max_future_Q}, Reward: {reward}")

        future_value = self.discount_rate * max_future_Q
        assert np.isscalar(future_value), "Future value must be a scalar"

        learning_term = self.learning_rate * (reward + future_value)
        assert np.isscalar(learning_term), "Learning term must be a scalar"

        new_Q = (1 - self.learning_rate) * current_Q + learning_term
        assert np.isscalar(new_Q), "new_Q must be a scalar"

        self.Q[state_index[0], state_index[1], agentIndex, action_index] = new_Q

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
        steps_number = []


        for i in range(0, self.numOfAgents):
            start = i * division
            end = (i + 1) * division
            #agentInterval =  (0.0, 0.25) # (0.0, 0.3333333333333333)  (0.3333333333333333, 0.6666666666666666) (0.6666666666666666, 1.0)
            agentInterval = (start, end)
            print(agentInterval)
            agentColor = self.colorList[i % len(self.colorList)]
            # Run the algorithm for each agent
            mySearch, effective_path, explored_cells, foundTheGoal = self.run_single_agent(agentInterval, i)


            self.concatenate_new_elements(explored, explored_cells)

            a = agent(self.maze,footprints=True,color=agentColor,shape='square',filled=True)
            
            
            def process_path(path):
                processed_path = []
                for cell in path:
                    if cell in processed_path:
                        # Find the first occurrence of the repeated cell and remove all cells up to that point
                        idx = processed_path.index(cell)
                        processed_path = processed_path[:idx]
                    processed_path.append(cell)
                return processed_path


            print(effective_path)
            paths.append({a:process_path(effective_path)})
            agents_search.append(mySearch)

            steps_number.append(len(process_path(effective_path)))

            if mySearch[-1] == (1,1):
                self._agent_done[i]= True
            
            # Number of steps of the agent. Subtract 1 to consider that the first cell is not countable
            agent_steps = len(mySearch) - 1

            # Count the total number of steps
            totalSteps += agent_steps

            # Get the number of the steps of the pionner
            if foundTheGoal == True:
                pionner_steps = agent_steps if pionner_steps > agent_steps else pionner_steps


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


        agent_path_dict = {}
        for d in paths:
            agent_path_dict.update(d)

        print(agent_path_dict)
       
       
        self.maze.tracePath(agent_path_dict, delay=200,kill=True)

       # self.maze.tracePath(paths[0], delay=200,kill=True)

        self.maze.run()
        print(self._agent_done)
        print("The steps taken by each agent was:", steps_number)
        return totalSteps, pionner_steps, fraction, fraction_pionner


    # Run the algorithm for a single agent
    def run_single_agent(self, agentInterval, agentIndex):
        explored = [self.start]
        mySearch = []
        start = self.start
        parentList = []
        parentList.append((-1,-1))
        currCell = self.start
        agent_path = []


        # Some agents will not find the goal because
        # currently the algorithm has a stop condition
        foundTheGoal = False
        while True:

            if currCell==self.maze._goal:
                mySearch.append(currCell)
                foundTheGoal = True
                break

            # If there are not non-visited children, go to parent
            nonVisitedChildren, allChildren = self.getChildrenPoints(currCell, self.maze.maze_map[currCell], parentList[-1], explored)
            
            count_nonVisitedChildren = len(nonVisitedChildren)
            if count_nonVisitedChildren == 0:
                if currCell not in explored:
                    explored.append(currCell)

                mySearch.append(currCell)

                # Stop condition
                if currCell == self.start:
                    break
                
                currCell = parentList.pop()
                agent_path.pop()

                continue

            # Check if the current cell belongs to another agent's interval
            Another_agent = self.does_currcell_belong_to_another_agent_interval(allChildren, agentIndex, agent_path)
            
            # Define the next step to the agent
            # If next == -1, go to parent
            next = self.defineAgentNextStep(agentInterval, agent_path, allChildren, nonVisitedChildren, currCell, agentIndex)
            
            if next == -1:
                if currCell not in explored:
                    explored.append(currCell)

                mySearch.append(currCell)

                if currCell != self.start:
                    currCell = parentList.pop()
                    agent_path.pop()

                continue

            elif next == None:
                break
            
            childCellPoint = allChildren[next]

            if currCell not in explored:
                explored.append(currCell)

            oldCell = copy.deepcopy(currCell) 

            parentList.append(currCell)
            mySearch.append(currCell)
            if childCellPoint=='N':
                currCell = (currCell[0]-1,currCell[1])
            elif childCellPoint=='E':
                currCell = (currCell[0],currCell[1]+1)
            elif childCellPoint=='S':
                currCell = (currCell[0]+1,currCell[1])     
            elif childCellPoint=='W':
                currCell = (currCell[0],currCell[1]-1)


            self.update_Q_before(oldCell, explored, agentIndex, childCellPoint, currCell, Another_agent)

        start = mySearch[-1]
        print(start)
        #print(explored)

        if foundTheGoal == False:
            # Beginning of Q-learning
            num_episodes = 30000
            exploration_rate = 1
            max_exploration_rate = 1
            min_exploration_rate = 0.01
            exploration_decay_rate =  0.0001
            max_steps_per_episode = 100
            effective_path = copy.deepcopy(mySearch) 

            for episode in range(num_episodes):
                q_path = [] 

                reward_current_episode = 0

                for step in range(max_steps_per_episode):
                    
                    action = self.choose_action(currCell, agentIndex, exploration_rate)
                    nextCell = self.get_next_state(currCell, action)
                    reward, is_terminal = self.calculate_reward(currCell, nextCell)

                    # Update Q-values
                    self.update_Q(currCell, action, reward, nextCell, agentIndex)


                    reward_current_episode += reward

                    # Stop if hit a wall (nextCell didn't change)
                    if nextCell == currCell:
                        print(f"Hit a wall at {nextCell} with action {action}, go back to {start}")
                        currCell = start
                        break

                    # Update the current cell and paths
                    currCell = nextCell
                    q_path.append(currCell)
                    mySearch.append(currCell)
                    explored.append(currCell)

                    if is_terminal:
                        foundTheGoal = True 
                        print("I found the goal")
                        break

                exploration_rate =  min_exploration_rate + \
                (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

                if is_terminal:
                    foundTheGoal = True 
                    break

            effective_path = effective_path + q_path    
            print("q_path", q_path)
        else: 
            print("already done")
            effective_path = mySearch
            
        return mySearch, effective_path, explored, foundTheGoal

    def calculate_reward(self, current, next_state):
        if current == next_state:  # No movement occurred, hit a wall
            return -1, False  # Negative reward for hitting a wall, not terminal
        if next_state == (1, 1):  # Check if it's the goal
            return 10, True  # Positive reward for reaching the goal, terminal
        return 0, False  # Standard move with no immediate reward

    

    def get_next_state(self, current, action):

        def check_movement_in_direction(maze, x, y, direction):
            # Check if the specified direction from the current position (x, y) is open (1) or blocked (0)
            # Ensure the cell exists in the maze dictionary and check the specific direction
            if (x, y) in maze:
                if direction in maze[(x, y)]:
                    return maze[(x, y)][direction]
                else:
                    return 0  # Assume blocked if the direction is not specified
            return 0  # Assume blocked if the cell is not in the maze dictionary
        
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


        movement_result = check_movement_in_direction(self.maze_map, x, y, action)
    
        if movement_result == 1:
            return next_state
        else:
            return current  # If calculated next state is still out of bounds, revert to current

    def does_currcell_belong_to_another_agent_interval(self, allChildren, agentIndex, agent_path):
        totalNumberOfChildren = len(allChildren)
        relative_node_weights = self.getRelativeNodeWeights(agent_path, totalNumberOfChildren)
        
        for i, interval in enumerate(self.agent_intervals):
            if i != agentIndex:
                for weight in relative_node_weights:
                    if interval[0] < weight[1] and interval[1] > weight[0]:
                        return True
        return False
    
    # Return children's cardinal points in preferential order 
    def getChildrenPoints(self, cellCoordinate, cellPoints, parent, explored):
        allChildren = []
        nonVisitedChildren = []
        for d in self.actions:
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
    
