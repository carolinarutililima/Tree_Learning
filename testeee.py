from pyamaze import maze, agent, textLabel
from threading import Thread
import time

def run_maze_with_delay(m, delay):
    m.run()
    time.sleep(delay)
    m.close()

m = maze(10, 10)
m.CreateMaze()
a = agent(m, filled=True, footprints=True)

delay = 10  # seconds
thread = Thread(target=run_maze_with_delay, args=(m, delay))
thread.start()
# thread.join()  # Uncomment this if you need the script to wait for the window to close
