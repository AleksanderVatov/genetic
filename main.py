import blackbox
import random

oracle = blackbox.BlackBox('shredded.png', 'original.png')
p = list(range(128))
random.shuffle(p)
print(oracle.evaluate_solution(p))  # Print the solution's fitness
# oracle.show_solution(p) # Display the solution
