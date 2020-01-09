from deap import algorithms, base, creator, tools

import blackbox
import random
import numpy as np
import matplotlib.pyplot as plt
import operator

# Call premade oracle
oracle = blackbox.BlackBox('shredded.png', 'original.png')
toolbox = base.Toolbox()

def evaluation(individual):
    return (oracle.evaluate_solution(individual),)


# Create fitness attribute to be minimized and assign it to the individual.
creator.create("FitnessMin", base.Fitness, weights=((-1.),))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("indices", np.random.permutation, 128)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=5e-3)

toolbox.register("evaluate", evaluation)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=100)

fit_stats = tools.Statistics(key=operator.attrgetter("fitness.values"))
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

result, log = algorithms.eaSimple(pop, toolbox,
                                cxpb=.9, mutpb=1,
                                ngen=500, stats=stats, verbose=False)

best_individual = tools.selBest(result, k=1)[0]
print('Fitness of the best individual: ', evaluation(best_individual)[0])

plt.figure(figsize=(11, 4))
plots = plt.plot(log.select('min'), 'c-', log.select('avg'), 'b-')
plt.legend(plots, ('Minimum fitness', 'Mean fitness'), frameon=True)
plt.ylabel('Fitness')
plt.xlabel('Iterations')

plt.show()


# Display the solution
# oracle.show_solution(order)