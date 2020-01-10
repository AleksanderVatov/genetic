from deap import algorithms, base, creator, tools

import blackbox
import random
import numpy as np
import matplotlib.pyplot as plt

# Evaluating function
def evaluation(individual):
    return (oracle.evaluate_solution(individual),)


# Call premade oracle
oracle = blackbox.BlackBox('shredded.png', 'original.png')
toolbox = base.Toolbox()

# Fit the creator and the toolbox
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("indices", random.sample, range(128), 128)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxUniformPartialyMatched,indpb=0.25)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.005)#5e-3)

toolbox.register("evaluate", evaluation)
toolbox.register("select", tools.selTournament, tournsize=3)

# Specify population
pop = toolbox.population(n=100)

# Add statistics
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


# Simulation runs
log_file = open('logs.txt', 'w')
mastertable = []

for s in range(1, 31):
    result, log = algorithms.eaSimple(pop, toolbox,
                                    cxpb=.9, mutpb=1,
                                    ngen=500, stats=stats, verbose=False)

    mastertable.append(log.select('avg'))
    mastertable.append(log.select('std'))
    mastertable.append(log.select('min'))
    mastertable.append(log.select('max'))

print(mastertable, file=log_file)


# Visualization
plt.figure(figsize=(11, 4))
plots = plt.plot(log.select('min'), 'c-', log.select('avg'), 'b-')
plt.legend(plots, ('Minimum fitness', 'Mean fitness'), frameon=True)
plt.ylabel('Fitness')
plt.xlabel('Generations')
plt.show()


# solution = [27, 23, 1, 21, 93, 13, 42, 41, 46, 35, 37, 84, 30, 8, 72, 76, 57, 64, 43, 20, 32, 82, 118, 114, 62, 70, 79, 47, 69, 108, 24, 125, 66, 97, 99, 54, 51, 12, 58, 4, 122, 104, 75, 45, 113, 56, 9, 115, 83, 0, 16, 87, 119, 65, 59, 17, 63, 121, 98, 53, 88, 10, 7, 28, 91, 14, 103, 44, 95, 49, 2, 26, 36, 110, 55, 109, 124, 86, 3, 
# 61, 11, 126, 111, 112, 106, 120, 81, 48, 105, 18, 116, 6, 68, 25, 80, 107, 89, 31, 96, 100, 29, 74, 102, 77, 
# 52, 22, 50, 71, 60, 67, 94, 92, 90, 40, 127, 5, 33, 34, 39, 78, 85, 123, 117, 38, 73, 19, 15, 101]

# # Display the solution
# oracle.show_solution(solution)