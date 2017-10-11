from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

BIRDS_COUNT = 20

try:
    xrange
except NameError:
    xrange = range

from sklearn.neural_network import MLPRegressor


import numpy as np
from random import shuffle
from itertools import accumulate, count
import math
import copy

class Bird:

    def __init__(self, brain = None):
        
        if brain is None:
            initDataX = [ [BASEY, 0], [-BASEY, 0] ]
            initDatay = [ 1, 0 ]

            self.brain = MLPRegressor(hidden_layer_sizes=(6, ), max_iter=10)
            self.brain.fit(initDataX, initDatay)

            self.brain.out_activation_ = "logistic"
        else:
            self.brain = copy.deepcopy(brain)

        self.score = 0

    def isJump(self, X):
        yPredict = self.brain.predict(X)
        return yPredict[0] > 0.5

    def decisionGenerator(self):
        count = 0
        dna = [0, 0, 1, 1]
        while True:
            if count <= 0:
                shuffle(dna)
                count = len(dna)
            count -= 1
            yield dna[count]

    def mutateDecisionGenerator(self):
        count = 0
        dna = [0, 0, 1]
        while True:
            if count <= 0:
                shuffle(dna)
                count = len(dna)
            count -= 1
            yield dna[count]

    # cross over
    def birth(self, other):
        son = Bird(self.brain)

        for ilyr in range(self.brain.n_layers_):
            for i in range(len(self.brain.coefs_)):
                for j in range(len(self.brain.coefs_[i])):
                    if next(self.decisionGenerator()):
                        son.brain.coefs_[i][j] = other.brain.coefs_[i][j]

        for ilyr in range(self.brain.n_layers_):
            for i in range(len(self.brain.intercepts_)):
                if next(self.decisionGenerator()):
                    son.brain.intercepts_[i] = other.brain.intercepts_[i]

        return son

    def mutate(self):
        for ilyr in range(self.brain.n_layers_):
            for i in range(len(self.brain.coefs_)):
                for j in range(len(self.brain.coefs_[i])):
                    if next(self.mutateDecisionGenerator()):
                        self.brain.coefs_[i][j] += random.randint(1, 10) / 10 * (next(self.decisionGenerator()) * 2 - 1)

        for ilyr in range(self.brain.n_layers_):
            for i in range(len(self.brain.intercepts_)):
                if next(self.mutateDecisionGenerator()):
                    self.brain.intercepts_[i] += random.randint(1, 10) / 10 * (next(self.decisionGenerator()) * 2 - 1)


class BirdsPopulation(list):

    def __init__(self, birdsCount = -1, birds = None):
        if (birdsCount < 0):
            self.birdsCount = len(birds)
        else:
            self.birdsCount = birdsCount
        if (birds is None):
            self.birds = [ Bird() for i in range(birdsCount) ]
        else:
            self.birds = birds

    def __getitem__(self, key):
        return self.birds[key]

    def __setitem__(self, key, item):
        self.birds[key] = item

    def split(self, ratioNumber, actualSize):
        totalRatioNumber = sum(ratioNumber)
        result = list(map(lambda x : math.floor(x / totalRatioNumber * actualSize), ratioNumber))
        result[-1] += actualSize - sum(result)
        return result
        
    def next(self):
        self.birds.sort(key=lambda x : -x.score)
        print('fitness function: ', end='')
        for bird in self.birds:
            print(bird.score, end=' ')
        print('\n')
        
        groupSplitNumber = self.split([30, 40, 30], self.birdsCount)
        groupStartNumber = [0] + list(accumulate(groupSplitNumber))[:-1]
        groupIndex = [ list(range(x, x+y)) for x, y in zip(groupStartNumber, groupSplitNumber) ]
        
        newPopulationCount = math.floor(self.birdsCount * 0.7)
        populationSplitNumber = self.split([4, 3, 1], newPopulationCount)
        crossing = [ [0, 0], [0, 1], [1, 2] ]
        
        newPopulation = []
        for (x, y), pop in zip(crossing, populationSplitNumber):
            iX, iY = BIRDS_COUNT + 1, BIRDS_COUNT + 1    
            for i in range(pop):
                if (iX >= groupSplitNumber[x]):
                    xList = list(groupIndex[x])
                    shuffle(xList)
                    iX = 0
                if (iY >= groupSplitNumber[y]):
                    yList = list(groupIndex[y])
                    shuffle(yList)
                    iY = 0
                newPopulation.append(self.birds[xList[iX]].birth(self.birds[yList[iY]]))
                iX += 1
                iY += 1
                
        shuffle(newPopulation)
        
        mutationCount = math.floor(self.birdsCount * 0.2)
        for i in range(mutationCount):
            newPopulation[i].mutate()
        newPopulation.extend(self.birds[:self.birdsCount - math.floor(self.birdsCount * 0.8)])
        for i in range(BIRDS_COUNT - len(newPopulation)):
            newBird = Bird(self.birds[i].brain)
            newBird.mutate()
            newPopulation.append(newBird)
            
        return BirdsPopulation(self.birdsCount, newPopulation)


class SoundEffectDump:
    
    def play(self):
        print(self.msg)

    def __init__(self, msg):
        self.msg = msg

def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = SoundEffectDump("die")
    SOUNDS['hit']    = SoundEffectDump("hit")
    SOUNDS['point']  = SoundEffectDump("point")
    SOUNDS['swoosh'] = SoundEffectDump("swoosh")
    SOUNDS['wing']   = SoundEffectDump("wing")

    while True:
        # select random background sprites
        randBg = 0
        #randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        #randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        randPlayer = 0
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        #pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        pipeindex = 0
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        global birds
        global lastScore
        global fastForward
        fastForward = False
        birds = BirdsPopulation(BIRDS_COUNT)
        generationCount = 1
        bestScore = 0
        while True:
            lastScore = 0
            print("=========== Generation - {} ===========".format(generationCount))
            movementInfo = initPosition()
            crashInfo = mainGame(movementInfo)
            bestScore = max(bestScore, lastScore)
            print("=========== Best Score - {} ===========".format(bestScore))
            birds = birds.next()
            generationCount += 1

        pygame.quit()
        sys.exit()
        #showGameOverScreen(crashInfo)


def initPosition():
    # index of player to blit on screen
    playerIndexGen = cycle([0, 1, 2, 1])

    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    basex = 0

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    return {
        'playery': playery + playerShmVals['val'],
        'basex': basex,
        'playerIndexGen': playerIndexGen,
    }

def mainGame(movementInfo):
    global birds
    global playerDied

    loopIter = 0
    score = [0] * BIRDS_COUNT
    playerIndex = [0] * BIRDS_COUNT
    playerIndexGen = [ movementInfo['playerIndexGen'] ] * BIRDS_COUNT
    playerx, playery = [ int(SCREENWIDTH * 0.2) ] * BIRDS_COUNT, [ movementInfo['playery'] ] * BIRDS_COUNT

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  [-9] * BIRDS_COUNT   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  [10] * BIRDS_COUNT   # max vel along Y, max descend speed
    playerMinVelY =  [-8] * BIRDS_COUNT   # min vel along Y, max ascend speed
    playerAccY    =  [ 1] * BIRDS_COUNT   # players downward accleration
    playerRot     =  [45] * BIRDS_COUNT   # player's rotation
    playerVelRot  =  [ 3] * BIRDS_COUNT   # angular speed
    playerRotThr  =  [20] * BIRDS_COUNT   # rotation threshold
    playerFlapAcc =  [-9] * BIRDS_COUNT   # players speed on flapping
    playerFlapped = [False] * BIRDS_COUNT # True when player flaps
    playerDied    = [False] * BIRDS_COUNT
    playersLeft   = BIRDS_COUNT
    
    global fastForward
    travelDistance = 0
    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE):
                fastForward = not fastForward
            """
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True
                    SOUNDS['wing'].play()
            """
        #print(playerx[0], (lowerPipes[1]['x'] if lowerPipes[0]['x'] + IMAGES['pipe'][0].get_width() < playerx[0] else lowerPipes[0]['x']) + IMAGES['pipe'][0].get_width())

        for i in range(BIRDS_COUNT):
            if playerDied[i]: continue

            
            upperX = lowerPipes[0]['x'] + IMAGES['pipe'][0].get_width()

            for j in count(0):
                if lowerPipes[j]['x'] + IMAGES['pipe'][0].get_width() < playerx[i]:
                    continue
                closeX = lowerPipes[j]['x'] + IMAGES['pipe'][0].get_width()
                centerGapY = (lowerPipes[j]['y'] - (PIPEGAPSIZE / 2))
                break

            data = [ [playery[i] - centerGapY, closeX - playerx[i]] ]

            wannaJump = birds[i].isJump(data)
            if wannaJump:
                if playery[i] > -2 * IMAGES['player'][0].get_height():
                    playerVelY[i] = playerFlapAcc[i]
                    playerFlapped[i] = True

            # check for crash here
            crashTest = checkCrash({'x': playerx[i], 'y': playery[i], 'index': playerIndex[i]},
                                   upperPipes, lowerPipes)
            if crashTest[0]:
                print('bird {} died with score {}'.format(i, score[i]))
                birds[i].score = travelDistance * 100 + BASEY - abs(playery[i] - centerGapY)
                playerDied[i] = True
                playersLeft -= 1
                if playersLeft == 0:
                    return {
                        'y': playery,
                        'groundCrash': crashTest[1],
                        'basex': basex,
                        'upperPipes': upperPipes,
                        'lowerPipes': lowerPipes,
                        'score': score,
                        'playerVelY': playerVelY,
                        'playerRot': playerRot
                    }

            # check for score
            playerMidPos = playerx[i] + IMAGES['player'][0].get_width() / 2
            for pipe in upperPipes:
                pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
                if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                    score[i] += 1
                    showScore(score[i])
                    #SOUNDS['point'].play()

            # playerIndex basex change
            if (loopIter + 1) % 3 == 0:
                playerIndex[i] = next(playerIndexGen[i])

        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        #showScore(score[i])


        for i in range(BIRDS_COUNT):
            if playerDied[i]: continue
            # rotate the player
            if playerRot[i] > -90:
                playerRot[i] -= playerVelRot[i]

            # player's movement
            if playerVelY[i] < playerMaxVelY[i] and not playerFlapped[i]:
                playerVelY[i] += playerAccY[i]
            if playerFlapped[i]:
                playerFlapped[i] = False

                # more rotation to cover the threshold (calculated in visible rotation)
                playerRot[i] = 45

            playerHeight = IMAGES['player'][playerIndex[i]].get_height()
            playery[i] += min(playerVelY[i], BASEY - playery[i] - playerHeight)

            # Player rotation has a threshold
            visibleRot = playerRotThr[i]
            if playerRot[i] <= playerRotThr[i]:
                visibleRot = playerRot[i]
            
            playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex[i]], visibleRot)
            SCREEN.blit(playerSurface, (playerx[i], playery[i]))

        travelDistance += 1

        pygame.display.update()
        if (not fastForward):
            FPSCLOCK.tick(FPS)


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]

lastScore = 0

def showScore(score):
    """displays score in center of screen"""
    """
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()
    """
    global lastScore
    global playerDied
    if lastScore < score:
        lastScore = score
        playerIdx = [ idx for idx, x in enumerate(playerDied) if not x ]
        print('current score = {}. still flapping = {}'.format(score, playerIdx))


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
