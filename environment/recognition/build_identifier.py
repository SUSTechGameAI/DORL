import gym_gvgai as gvg
from identifier import Identifier
import identifier
# golddigger treasurekeeper waterpuzzle
gameName = 'golddigger'

if __name__ == '__main__':
    idfy = identifier.buildIdentifier(gameName, ['lvl0'])
    idfy.save('./identifier/%s.json' % gameName)