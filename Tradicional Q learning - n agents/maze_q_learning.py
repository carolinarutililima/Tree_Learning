import gym

class Maze(gym.Env ):

    def __init__(self, maze_map, numOfLines, numOfColumns):
        self.MAZE_C = numOfColumns
        self.MAZE_R = numOfLines
        self.map_maze = maze_map


    def maze_mov(self, action, x, y):
        
        print(x,y)
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
            x_new = x-1
            y_new = y
        elif action == 1:   # down
            direction = 'S'
            x_new = x+1
            y_new = y                       
        elif action == 2:   # right
            direction = 'E'
            x_new = x
            y_new = y+1  
        elif action == 3:   # left
            direction = 'W'
            x_new = x 
            y_new = y-1   

        print("new", x_new, y_new)

        movement_result = check_movement_in_direction(self.map_maze, x, y, direction)
        if movement_result == 1:
            x, y = x_new,y_new
        else:
            x,y = (self.MAZE_C,self.MAZE_R)

        print("actual postion", x,y)
        return (x,y)




    def reset(self):
        return (self.MAZE_C, self.MAZE_R)
    

    def step(self, action, x, y):
        print("The action was:", action)
        print("current position", x,y)
        done = False 
        current_value = self.maze_mov(action, x, y)

        print("valor atual p reward", current_value)
        # reward function
        if current_value == (self.MAZE_C,self.MAZE_R):
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