import blackbox
import deap             # pip install deap

oracle = blackbox.BlackBox('shredded.png')
p = range(64)
oracle.show_solution(p)





if __name__ == '__main__':
    print('Running program...')

    print(dir(blackbox))