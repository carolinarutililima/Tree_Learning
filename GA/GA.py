import matplotlib.pyplot as plt
import random,time
from pyamaze import maze,agent,COLOR


def display(lst):
    end=time.time()
    print(lst)
    a=agent(m,footprints=True)

    m.tracePath({a:lst})
    # m.tracePath({b:m.path},delay=10)
    best_in_pop.append(0)
    print(f'\n\nsolution occured in generation : {generation+1}')
    print(f'time taken to find solution is : {end-startt} seconds')
    
    plt.plot(best_in_pop)
    plt.xlabel('generations')
    plt.ylabel('min.fitness value per generation')
    plt.title(f'MazeSize={row},{column}             time taken to find solution={(end-startt):.2f}sec.\nFitness vs Generations using PopulationSize of {pop_size}\n')
    m.run(),plt.show()
    quit()



def crossover(population): 
    'this function will crossover the population and will return the crossoverd population'
    goodones=population[:]
    for i in range(0,len(population),2):
        crossoverpoint=random.randint(1,column-1)
        # print(f'crossoverpoint is {crossoverpoint}')
        population[i][crossoverpoint:],population[i+1][crossoverpoint:]=population[i+1][crossoverpoint:],population[i][crossoverpoint:]
    return goodones+population
#example:
#parents=[
#     [(7, 7), (7, 6), (6, 5), (5, 4), (6, 3), (6, 2), (1, 1)], 
#     [(7, 7), (7, 6), (6, 5), (5, 4), (5, 3), (7, 2), (1, 1)]]
# crossoverpoint=4
# offsprinfs=
#   [(7, 7), (7, 6), (6, 5), (5, 4), (5, 3), (7, 2), (1, 1)], 
#   [(7, 7), (7, 6), (6, 5), (5, 4), (6, 3), (6, 2), (1, 1)]]

def mutation(population): 
    'this function will mutate the population and will return the mutated population'
    for path in population:
        mutationpoint=random.randint(1,column-1)
        # print(f'mutationpoint is {mutationpoint}')
        path[mutationpoint]=(random.randint(1,row),path[mutationpoint][1])
    return population

# sampple chromosome=[(7, 7), (7, 6), (4, 5), (5, 4), (7, 3), (6, 2), (1, 1)]
# mutationpoint=5
# mutated chromosome=[(7, 7), (7, 6)

def sort_select(population):
    'istead of defining a separate sorting , we done it in the fitness function directly'
    fitness_lst=[path[-1] for path in population]
    best_in_pop.append(min(fitness_lst))
    population=[path[:-1] for path in sorted(population,key=lambda x:x[-1])]
    return population[:int(len(population)/2)]




def fitness_function(population):#2
    'this function will generate the row-first and column- first path from the chromosome and will return the shortest of them'
    for chromosome in population:
        col_fitness=0
        row_fitness=0
        chromosome[0]=start
        chromosome[-1]=goal
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

        if col_fitness==0:
             
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
        if row_fitness==0:
            display(row_first_path)

        # print(f'row_fitness is {row_fitness} and col_fitness is {col_fitness}')
        if col_fitness<=row_fitness:
                chromosome.append(col_fitness)
        else:
                chromosome.append(row_fitness) 
    return population


def population_generator(row,column,popsize):#1
    'this function will generate the population(which is a list of paths) and the length of the population is popsize)'
    population=[[(random.randint(1,start[0]),x) for x in range(start[1],0,-1)] for _ in range(popsize)]
    # print(f'After generation, population is :\n{population}\n\n\n')
    return population
#main program

#_____________________________________________________________________________________
#default values
row,column=10,10
pop_size=500 #population size is only mulitple of 100 because the crossover operation require pairs of parents
max_gen=500

start=(row,column)
goal=(1,1)
# # _____________________________________________________________________________________

m=maze(row,column)
m.CreateMaze(loopPercent=100)
mazedict=m.maze_map
# # ______________________________________________________________________________________

generation=0
startt=time.time()
population=population_generator(row,column,pop_size)
best_in_pop=[]
while True:
    population=mutation(crossover(sort_select((fitness_function(population)))))
    generation+=1
    if generation==max_gen:
        print('Sorry! No Solution Found')
        plt.plot(best_in_pop)
        plt.xlabel('\ngenerations')
        plt.ylabel('min.fitness value per generation\n')
        plt.title(f'MazeSize={row},{column}\n____Sorry! Solution not found_____\nFitness vs Generations using PopulationSize of {pop_size}\n')
        m.run()
        plt.show()
        quit()