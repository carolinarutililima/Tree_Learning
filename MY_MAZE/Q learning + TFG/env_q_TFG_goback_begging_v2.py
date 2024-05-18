
import numpy as np
from pyamaze import maze, agent, COLOR
import sys
import random

class MyAlgorithm:
    def __init__(self, maze, n_agents, colorList, start=None):
        self.maze = maze
        self.numOfAgents = n_agents
        self.colorList = colorList
        self.start = start if start is not None else (maze.rows, maze.cols)
        self.Q = np.zeros((maze.rows, maze.cols, n_agents, 4))  # Initialize Q-table with dimensions for each cell and action
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.actions = ['N', 'E', 'S', 'W']  # Possible actions
        self.compass = ['N', 'E', 'S', 'W']  # Defining compass to be used in getChildrenPoints
        self.filledInterval = [False for _ in range(n_agents)]

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

    # Run the algorithm for all agents
    def run(self):
        division = 1.0 / self.numOfAgents
        paths = []
        explored = []
        agents_search = []
        pionner_steps = sys.maxsize
        totalSteps = 0
        for i in range(self.numOfAgents):
            start = i * division
            end = (i + 1) * division
            agentInterval = (0.5, 1.0)
            agentColor = self.colorList[i % len(self.colorList)]

            # Run the algorithm for each agent
            mySearch, effective_path, explored_cells, foundTheGoal = self.run_single_agent(agentInterval, i)

            self.concatenate_new_elements(explored, explored_cells)

            a = agent(self.maze, footprints=True, color=agentColor, shape='square', filled=True)

            paths.append({a: mySearch})
            agents_search.append(mySearch)

            # Number of steps of the agent. Subtract 1 to consider that the first cell is not countable
            agent_steps = len(mySearch) - 1

            # Count the total number of steps
            totalSteps += agent_steps

            # Get the number of the steps of the pionner
            if foundTheGoal:
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

    # Run the algorithm for a single agent
    def run_single_agent(self, agentInterval, agentIndex):
        mySearch = []
        effective_path = []
        explored = []
        foundTheGoal = False

        for episode in range(1000):  # Set a limit for episodes to prevent infinite loops
            currCell = self.start
            parentList = []
            parentList.append((-1, -1))
            agent_path = []

            while True:
                if currCell == self.maze._goal:
                    mySearch.append(currCell)
                    effective_path.append(currCell)
                    foundTheGoal = True
                    break

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
                    break

            # If goal found, stop the episode
            if foundTheGoal:
                break

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
            if cellPoints[d]:
                if d == 'N':
                    childCell = (cellCoordinate[0] - 1, cellCoordinate[1])
                    if parent == childCell:
                        continue
                    if childCell in explored:
                        allChildren.append('N')
                        continue

                    allChildren.append('N')
                    nonVisitedChildren.append('N')
                elif d == 'E':
                    childCell = (cellCoordinate[0], cellCoordinate[1] + 1)
                    if parent == childCell:
                        continue
                    if childCell in explored:
                        allChildren.append('E')
                        continue

                    allChildren.append('E')
                    nonVisitedChildren.append('E')
                elif d == 'S':
                    childCell = (cellCoordinate[0] + 1, cellCoordinate[1])
                    if parent == childCell:
                        continue
                    if childCell in explored:
                        allChildren.append('S')
                        continue

                    allChildren.append('S')
                    nonVisitedChildren.append('S')
                elif d == 'W':
                    childCell = (cellCoordinate[0], cellCoordinate[1] - 1)
                    if parent == childCell:
                        continue