import tkinter as tk
import random

# Constants
SQUARE_SIZE = 20
MAZE_SIZE = 10  # Adjust the maze size as needed
ANT_COUNT = 10
ANT_SPEED = 200  # Milliseconds per ant move

# Colors
EMPTY_COLOR = "white"
WALL_COLOR = "black"
ANT_COLOR = "red"
FOOD_COLOR = "green"
BORDER_COLOR = "gray"


class Ant:
    def __init__(self, canvas, start_x, start_y):
        self.canvas = canvas
        self.x = start_x
        self.y = start_y
        self.id = canvas.create_oval(
            start_x, start_y, start_x + SQUARE_SIZE, start_y + SQUARE_SIZE, fill=ANT_COLOR)
        self.directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def move(self):
        dx, dy = random.choice(self.directions)
        new_x = self.x + dx * SQUARE_SIZE
        new_y = self.y + dy * SQUARE_SIZE

        if 0 <= new_x < MAZE_SIZE * SQUARE_SIZE and 0 <= new_y < MAZE_SIZE * SQUARE_SIZE:
            self.canvas.move(self.id, dx * SQUARE_SIZE, dy * SQUARE_SIZE)
            self.x = new_x
            self.y = new_y


class MazeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ant Exploration")
        self.canvas = tk.Canvas(
            root, width=MAZE_SIZE * SQUARE_SIZE, height=MAZE_SIZE * SQUARE_SIZE)
        self.canvas.pack()
        self.maze = [[0 for _ in range(MAZE_SIZE)] for _ in range(MAZE_SIZE)]
        self.generate_maze()
        self.ants = [Ant(self.canvas, 0, 0) for _ in range(ANT_COUNT)]
        self.food_x, self.food_y = self.place_food()
        self.food_counter = 0  # Initialize food counter
        self.update_ants()

    def generate_maze(self):
        # Keep everything free, no obstacles
        for i in range(MAZE_SIZE):
            for j in range(MAZE_SIZE):
                self.maze[i][j] = 0
                # Draw squares without borders
                self.canvas.create_rectangle(
                    j * SQUARE_SIZE, i *
                    SQUARE_SIZE, (j + 1) * SQUARE_SIZE, (i + 1) * SQUARE_SIZE,
                    fill=EMPTY_COLOR, outline=BORDER_COLOR)

    def place_food(self):
        while True:
            x = random.randint(0, MAZE_SIZE - 1) * SQUARE_SIZE
            y = random.randint(0, MAZE_SIZE - 1) * SQUARE_SIZE
            if self.maze[y // SQUARE_SIZE][x // SQUARE_SIZE] == 0:
                self.canvas.create_rectangle(
                    x, y, x + SQUARE_SIZE, y + SQUARE_SIZE, fill=FOOD_COLOR)
                return x, y

    def update_ants(self):
        for ant in self.ants:
            ant.move()
            if (ant.x, ant.y) == (self.food_x, self.food_y):
                # Change the color of the ant to dark green
                self.canvas.itemconfig(ant.id, fill="dark green")

                # Convert the square where the food is located to pink
                food_square_x = self.food_x // SQUARE_SIZE
                food_square_y = self.food_y // SQUARE_SIZE
                # Mark as food square
                self.maze[food_square_y][food_square_x] = 2

                # Update food counter and display in the terminal
                self.food_counter += 1
                print(f"Food Counter: {self.food_counter}")

                self.canvas.create_rectangle(
                    self.food_x, self.food_y, self.food_x + SQUARE_SIZE, self.food_y + SQUARE_SIZE, fill="pink")

                self.ants.remove(ant)
                if not self.ants:
                    print("All ants have found the food!")
                    return
        self.root.after(ANT_SPEED, self.update_ants)


def main():
    root = tk.Tk()
    gui = MazeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
