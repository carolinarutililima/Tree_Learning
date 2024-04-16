import matplotlib.pyplot as plt
import random,time
from pyamaze import maze,agent,COLOR




def display(lst):
    end=time.time()
    print(lst)
    a=agent(m,footprints=True)
    #m.tracePaths({a:lst},delay=10)
    # m.tracePath({b:m.path},delay=10)
    print(f'\n\nsolution occured in generation : {generation+1}')
    print(f'time taken to find solution is : {end-startt} seconds')
    m.run()
    plt.plot(best_in_pop)
    plt.xlabel('generations')
    plt.ylabel('min.fitness value per generation')
    plt.title(f'min.Fitness vs Generations using PopulationSize of {pop_size}\n(minimum fitness is best) \n')
    plt.show()

def population_generator(row,column,popsize):#1
    'this function will generate the population(which is a list of chromosomes) and the length of the population is popsize)'
    population=[[(random.randint(1,row),x) for x in range(column,0,-1)] for _ in range(popsize)]
    # print(f'After generation, population is :\n{population}\n\n\n')
    return population

def fitness_function(population):#2
    'this function will generate the row-first and column- first path and find fitness of each and will return the minimum'
    for chromosome in population:
        col_fitness=0
        row_fitness=0
        chromosome[0]=start  # defined in main block
        chromosome[-1]=goal  # defined in main block
        row_first_path=[start] 
        col_first_path=[start] 
        for i in range(1,len(chromosome)):
            while col_first_path[-1][0]>chromosome[i][0]:
                col_first_path.append((col_first_path[-1][0]-1,col_first_path[-1][1]))
                if m.maze_map[(col_first_path[-1][0],col_first_path[-1][1])]['S']==0:
                    col_fitness+=1
            while col_first_path[-1][0]<chromosome[i][0]:
                col_first_path.append((col_first_path[-1][0]+1,col_first_path[-1][1]))
                if m.maze_map[(col_first_path[-1][0],col_first_path[-1][1])]['N']==0:
                    col_fitness+=1
            while col_first_path[-1][1]>chromosome[i][1]:
                col_first_path.append((col_first_path[-1][0],col_first_path[-1][1]-1))
                if m.maze_map[(col_first_path[-1][0],col_first_path[-1][1])]['E']==0:
                    col_fitness+=1
            while col_first_path[-1][1]<chromosome[i][1]:
                col_first_path.append((col_first_path[-1][0],col_first_path[-1][1]+1))
                if m.maze_map[(col_first_path[-1][0],col_first_path[-1][1])]['W']==0:
                    col_fitness+=1

        if col_fitness==0: #no walls crossed. solution found
             display(col_first_path)

        for i in range(1,len(chromosome)):
                while row_first_path[-1][1]>chromosome[i][1]:
                    row_first_path.append((row_first_path[-1][0],row_first_path[-1][1]-1))
                    if m.maze_map[(row_first_path[-1][0],row_first_path[-1][1])]['E']==0:
                        row_fitness+=1
                while row_first_path[-1][1]<chromosome[i][1]:
                    row_first_path.append((row_first_path[-1][0],row_first_path[-1][1]+1))
                    if m.maze_map[(row_first_path[-1][0],row_first_path[-1][1])]['w']==0:
                        row_fitness+=1
                while row_first_path[-1][0]>chromosome[i][0]:
                    row_first_path.append((row_first_path[-1][0]-1,row_first_path[-1][1]))
                    if m.maze_map[(row_first_path[-1][0],row_first_path[-1][1])]['S']==0:
                        row_fitness+=1
                while row_first_path[-1][0]<chromosome[i][0]:
                    row_first_path.append((row_first_path[-1][0]+1,row_first_path[-1][1]))
                    if m.maze_map[(row_first_path[-1][0],row_first_path[-1][1])]['N']==0:
                        row_fitness+=1 

        if row_fitness==0: #no walls crossed. solution found
            display(row_first_path)

        # print(f'row_fitness is {row_fitness} and col_fitness is {col_fitness}')
        if col_fitness<=row_fitness:
                chromosome.append(col_fitness)
        else:
                chromosome.append(row_fitness) 
    print(population)
    return population

def sort_select(population):
    """
    This function sorts a population list based on fitness values assumed to be the last element in each sublist.
    It then returns a new population of the same size by duplicating the better half more frequently.

    :param population: List of lists, where each sublist represents an individual and the last element of each sublist is the fitness value.
    :return: A new population of the same size, with better-performing individuals duplicated.
    """
    # Check if the population is empty to avoid errors
    if not population:
        return []

    # Extracting fitness values for each individual
    fitness_lst = [path[-1] for path in population]

    # Logging the best fitness in the population (optional, if you want to track this outside the function)
    best_fitness = min(fitness_lst)
    best_in_pop.append(best_fitness)  # Assumes best_in_pop is defined outside this function as a list

    # Sorting the population based on the fitness value
    population_sorted = sorted(population, key=lambda x: x[-1])

    # Creating a new population of the same size by duplicating better performers
    new_population = []
    half_population_size = len(population) // 2
    for i in range(len(population)):
        # Duplicate individuals from the better half more frequently
        index = i % half_population_size  # This ensures that the better half has a higher chance of duplication
        new_population.append(population_sorted[index][:-1])  # Append without fitness value

    return new_population




def crossover(population):
    'this function will crossover the population and will return the crossoverd population'
    for i in range(0, len(population) - (len(population) % 2), 2):  # Ensures no out of range index
        crossoverpoint = random.randint(1, numOfColumns - 1)  # Assumes numOfColumns is defined elsewhere
        # Perform crossover
        population[i][crossoverpoint:], population[i+1][crossoverpoint:] = population[i+1][crossoverpoint:], population[i][crossoverpoint:]

    return population  # Returns only the modified population



def mutation(population ): 
    'this function will mutate the population and will return the mutated population'
    for path in population:
        mutationpoint=random.randint(1,numOfColumns-1)
        # print(f'mutationpoint is {mutationpoint}')
        path[mutationpoint]=(random.randint(1,numOfLines),path[mutationpoint][1])
    return population

# Size of the maze
numOfLines = 10
numOfColumns = 10
pop_size = 500
start = (numOfLines, numOfColumns)
goal = (1,1)

m=maze(numOfLines, numOfColumns)

# Create the maze 
#m.CreateMaze(theme=COLOR.light, saveMaze='Mazev1')
m.CreateMaze(theme=COLOR.light, loadMaze='Mazev1.csv')


# Load Maze
a = agent(m, numOfLines, numOfColumns, filled=False, shape= 'square')


best_in_pop = []
generation=0
startt=time.time()
population= population_generator(numOfLines, numOfColumns, pop_size)
max_gen = 500

while True:
    population=mutation(crossover(sort_select((fitness_function(population)))))
    generation+=1
    if generation==max_gen:
        print('Sorry! No Solution Found')
        plt.plot(best_in_pop)
        plt.xlabel('\ngenerations')
        plt.ylabel('min.fitness value per generation\n')
        plt.title(f'MazeSize={numOfLines},{numOfColumns}\n____Sorry! Solution not found_____\nFitness vs Generations using PopulationSize of {pop_size}\n')
        m.run()
        plt.show()
       
        m.run()
        quit()





