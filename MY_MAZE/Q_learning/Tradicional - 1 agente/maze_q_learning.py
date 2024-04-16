import gym





class Maze(gym.Env ):

    def __init__(self, maze_map, numOfLines, numOfColumns):
        self.MAZE_C = numOfColumns
        self.MAZE_R = numOfLines
        self.position = (self.MAZE_C, self.MAZE_R)
        self.map_maze = maze_map


    def maze_mov(self, action):

        def check_movement_in_direction(maze, x, y, direction):
            # Check if the specified direction from the current position (x, y) is open (1) or blocked (0)
            # Ensure the cell exists in the maze dictionary and check the specific direction
            if (x, y) in maze:
                if direction in maze[(x, y)]:
                    return maze[(x, y)][direction]
                else:
                    return 0  # Assume blocked if the direction is not specified
            return 0  # Assume blocked if the cell is not in the maze dictionary

        if action == 0:   # up
            direction = 'N'
            x_new = self.position[0]-1
            y_new = self.position[1]
        elif action == 1:   # down
            direction = 'S'
            x_new = self.position[0]+1
            y_new = self.position[1]                       
        elif action == 2:   # right
            direction = 'E'
            x_new = self.position[0]
            y_new = self.position[1]+1  
        elif action == 3:   # left
            direction = 'W'
            x_new = self.position[0] 
            y_new = self.position[1]-1   

        x = self.position[0]
        y = self.position[1]

        movement_result = check_movement_in_direction(self.map_maze, x, y, direction)
        if movement_result == 1:
            self.position = (x_new,y_new)
        else:
            self.position = (self.MAZE_C+1,self.MAZE_R+1)

        print("actual postion", self.position)
        return self.position




    def reset(self):
        self.position =  (self.MAZE_C,self.MAZE_R)
        return self.position
    

    def step(self, action):
        print("The action was:", action)
        done = False 
        current_value = self.maze_mov(action)

        print("valor atual p reward", current_value)
        # reward function
        if current_value == (self.MAZE_C+1,self.MAZE_R+1):
            # hit the wall
            reward = -1
            done = True

        elif current_value == (1,1):
            # our gold goal!
            reward = 1
            done = True 

        else:
            # just moving around!
            reward = 0
            done = False

        return current_value, reward, done, {}