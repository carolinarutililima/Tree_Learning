from pyamaze import maze, agent, textLabel
import tkinter as tk

def run_maze_with_auto_close(m, delay):
    # Schedule the window to close after 'delay' milliseconds
    m._win.after(delay, m._win.destroy)
    m.run()

# Create and configure the maze
m = maze(10, 10)
m.CreateMaze()
a = agent(m, filled=True, footprints=True)

# Define the delay in milliseconds (e.g., 10000ms for 10 seconds)
delay = 5000

# Run the maze and schedule the window to close
run_maze_with_auto_close(m, delay)

print('ok')
