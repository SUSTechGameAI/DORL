from identifier import Identifier
import gym_gvgai as gvg
import time
import numpy as np

gameName = 'waterpuzzle'

if __name__ == '__main__':
    idfy = Identifier()
    idfy.load('./identifier/%s.json' % gameName)
    playerCode = idfy.playerCode
    env = gvg.make('gvgai-%s-lvl0-v0' % gameName)
    pixels = env.reset()

    start = time.process_time_ns()
    res = idfy.identify(pixels)
    print(time.process_time_ns()-start)
    for i in range(len(res)):
        for j in range(len(res[0])):
            if res[i][j] == playerCode:
                print('A', end=' ')
                continue
            print(res[i][j], end=' ')
        print()
