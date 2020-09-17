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

BIRDS_COUNT = 21
MAX_ITEA = 120
MAX_SCORE = 1000
PM = 0.5
dump_file = f"iter-MAX_ITEA.plot"

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
import pickle
from pathlib import Path

class Bird:

    def __init__(self, brain = None):

        if brain is None:
            # featurenya: diff y player dgn pipe terdekat, jarak dgn piper terdekat, diff y pipe terdekat dgn pipe berikutnya
            initDataX = [ [BASEY / 2, 0, BASEY / 2], [-BASEY / 2, 0, -BASEY / 2] ]
            initDatay = [ 1, 0 ]

            self.brain = MLPRegressor(hidden_layer_sizes=(8, ), max_iter=10)
            self.brain.fit(initDataX, initDatay)

            self.brain.out_activation_ = "logistic"
        else:
            self.brain = copy.deepcopy(brain)

        self.score = 0

    def isJump(self, X):
        yPredict = self.brain.predict(X)
        return yPredict[0] > 0.5

    # cross over
    def cross_over(self, other):
        for i in range(self.brain.n_layers_ - 1):
            for j in range(len(self.brain.coefs_[i])):
                for k in range(len(self.brain.coefs_[i][j])):
                    if random.uniform(0, 1) <= .5:
                        self.brain.coefs_[i][j][k], other.brain.coefs_[i][j][k] = other.brain.coefs_[i][j][k], self.brain.coefs_[i][j][k]

        for i in range(self.brain.n_layers_ - 1):
            for j in range(len(self.brain.intercepts_[i])):
                if random.uniform(0, 1) <= .5:
                        self.brain.intercepts_[i][j], other.brain.intercepts_[i][j] = other.brain.intercepts_[i][j], self.brain.intercepts_[i][j]

    def mutate(self, pm):
        for i in range(self.brain.n_layers_ - 1):
            for j in range(len(self.brain.coefs_[i])):
                for k in range(len(self.brain.coefs_[i][j])):
                    if random.uniform(0, 1) <= pm:
                        self.brain.coefs_[i][j][k] += random.randint(1, 10) / 100 * (round(random.uniform(0, 1)) * 2 - 1)

        for i in range(self.brain.n_layers_ - 1):
            for j in range(len(self.brain.intercepts_[i])):
                if random.uniform(0, 1) <= pm:
                    self.brain.intercepts_[i][j] += random.randint(1, 10) / 100 * (round(random.uniform(0, 1)) * 2 - 1)

class BirdsPopulation(list):

    def __init__(self, birdsCount = -1, birds = None, generation_counts = 1):
        if (birdsCount < 0):
            self.birdsCount = len(birds)
        else:
            self.birdsCount = birdsCount
        if (birds is None):
            self.birds = [ Bird() for i in range(birdsCount) ]
        else:
            self.birds = birds

        self.generation_counts = generation_counts


    def __getitem__(self, key):
        return self.birds[key]

    def __setitem__(self, key, item):
        self.birds[key] = item

    def next(self):
        self.birds.sort(key=lambda x : -x.score)
        self.best_one = self.birds[0]
        print('fitness function: ', end='')
        for bird in self.birds:
            print(bird.score, end=' ')
        print('\n')

        # roulette
        sum_scores = [int(bird.score) if bird.score > 0 else 0 for bird in self.birds]
        for idx in range(1, len(sum_scores)):
            sum_scores[idx] += sum_scores[idx - 1]

        newPopulation = []
        population_idx = []
        for bird in range(self.birdsCount - 1):
            dart = random.randint(1, sum_scores[-1])
            for idx, sum_score in enumerate(sum_scores):
                if dart <= sum_score:
                    break
            newPopulation.append(copy.deepcopy(self.birds[idx]))
            population_idx.append(idx)

        # cross_over
        for idx in range(0, len(newPopulation) - 1, 2):
            # print(newPopulation[idx].brain.coefs_)
            # print(newPopulation[idx + 1].brain.coefs_)
            newPopulation[idx].cross_over(newPopulation[idx + 1])

        # mutate
        # print(PM * (MAX_ITEA - self.generation_counts) / MAX_ITEA)
        for new_bird in newPopulation:
            new_bird.mutate(PM * (MAX_ITEA - self.generation_counts) / MAX_ITEA)

        # add best bird of last population
        newPopulation.append(self.best_one)

        return BirdsPopulation(self.birdsCount, newPopulation, self.generation_counts + 1)

def saveConfig(bp):
    fout = open(f'latest_birds_config.txt', 'wb')
    pickle.dump(bp, fout)

def loadConfig():
    my_file = Path(f'latest_birds_config.txt')
    if my_file.exists():
        fin = open(f'latest_birds_config.txt', 'rb')
        return pickle.load(fin)
    return None

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
    # base (ground) sprite 地面的移动效果
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
    # pipeindex = random.randint(0, len(PIPES_LIST) - 1)
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
    birds = loadConfig()
    if (birds is None):
        birds = BirdsPopulation(BIRDS_COUNT)

    bestScore = 0
    # while True:
    best_score_each_iter = [0] * MAX_ITEA
    for iter in range(MAX_ITEA):
        saveConfig(birds)
        # update each iteration
        lastScore = 0
        print("=========== Generation - {} ===========".format(birds.generation_counts))
        movementInfo = initPosition()
        crashInfo = mainGame(movementInfo)
        bestScore = max(bestScore, lastScore)
        print("=========== Best Score - {} ===========".format(bestScore))
        if bestScore == MAX_SCORE:
            best_score_each_iter[iter:MAX_ITEA] = [lastScore] * (MAX_ITEA - iter)
            break
        birds = birds.next()

    with open(f"iter-{MAX_ITEA}.txt", 'wb') as f:
        pickle.dump(dump_file)
    pygame.quit()
    sys.exit()
    #showGameOverScreen(crashInfo)


def initPosition():
    # index of player to blit on screen
    playerIndexGen = cycle([0, 1, 2, 1])

    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    basex = 0

    # player shm for up-down motion on welcome screen
    # val: 小鸟初始位置 dir: 小鸟每次 up-down 的像素
    playerShmVals = {'val': 0, 'dir': 1}

    return {
        'playery': playery + playerShmVals['val'],
        'basex': basex,
        'playerIndexGen': playerIndexGen,
    }

def yScoreFunc(playerY, lGapY, uGapY):
    if lGapY <= playerY <= uGapY:
        return 0
    if (playerY < lGapY):
        return lGapY - playerY
    if (playerY > uGapY):
        return playerY - uGapY
    return 0

def mainGame(movementInfo):
    global birds
    global playerDied
    global lastScore

    loopIter = 0
    score = [0] * BIRDS_COUNT
    playerIndex = [0] * BIRDS_COUNT
    playerIndexGen = [ movementInfo['playerIndexGen'] ] * BIRDS_COUNT
    playerx, playery = [ int(SCREENWIDTH * 0.2) ] * BIRDS_COUNT, [ movementInfo['playery'] ] * BIRDS_COUNT

    basex = movementInfo['basex']
    # base（地面）比 background（背景）要长，移动效果就是 base 一直向左移动
    # baseShift 就是 base 可以向左移动的最大长度，不然就会出现黑色空洞
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lower pipes
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
        # print(playerx[0], (lowerPipes[1]['x'] if lowerPipes[0]['x'] + IMAGES['pipe'][0].get_width() < playerx[0] else lowerPipes[0]['x']) + IMAGES['pipe'][0].get_width())

        for i in range(BIRDS_COUNT):
            if playerDied[i]: continue


            upperX = lowerPipes[0]['x'] + IMAGES['pipe'][0].get_width()

            # 小鸟与当前即将通过的管道的水平距离（管道右侧）
            closeX = SCREENWIDTH
            # 小鸟与当前即将通过的管道的gap中心所在的高度
            centerGapY = SCREENHEIGHT / 2
            # 小鸟与下一个即将通过的管道的gap中心所在的高度
            nextCenterGapY = 0
            for j in range(len(lowerPipes)):
                if lowerPipes[j]['x'] + IMAGES['pipe'][0].get_width() < playerx[i]:
                    continue
                if lowerPipes[j]['x'] > SCREENWIDTH:
                    continue
                closeX = lowerPipes[j]['x'] + IMAGES['pipe'][0].get_width()
                centerGapY = (lowerPipes[j]['y'] - (PIPEGAPSIZE / 2))
                if j < len(lowerPipes) and j + 1 < len(lowerPipes) and lowerPipes[j + 1]['x'] <= SCREENWIDTH:
                    nextCenterGapY = (lowerPipes[j + 1]['y'] - (PIPEGAPSIZE / 2))
                break

            #if i == 0:
                #print(closeX, centerGapY, nextCenterGapY)
                #print(nextCloseX, nextCenterGapY)

            data = [ [playery[i] - centerGapY, closeX - playerx[i], centerGapY - nextCenterGapY] ]

            wannaJump = birds[i].isJump(data)
            if wannaJump:
                if playery[i] > -2 * IMAGES['player'][0].get_height():
                    playerVelY[i] = playerFlapAcc[i]
                    playerFlapped[i] = True

            # check for crash here
            crashTest = checkCrash({'x': playerx[i], 'y': playery[i], 'index': playerIndex[i]},
                                   upperPipes, lowerPipes)
            if crashTest[0]:
                playerHeight = IMAGES['player'][playerIndex[i]].get_height()

                print('bird {} died with score {}'.format(i, score[i]))
                birds[i].score = travelDistance * 100 - yScoreFunc(playery[i] + playerHeight / 2, centerGapY - (PIPEGAPSIZE / 2), centerGapY + (PIPEGAPSIZE / 2))
                playerDied[i] = True
                playersLeft -= 1
                if playersLeft == 0 or lastScore == MAX_SCORE:
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

            # check for score of each bird
            playerMidPos = playerx[i] + IMAGES['player'][0].get_width() / 2
            for pipe in upperPipes:
                pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
                if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                    score[i] += 1
                    if lastScore < score[i]:
                        lastScore = score[i]
                        playerIdx = [ idx for idx, x in enumerate(playerDied) if not x ]
                        print('current score = {}. still flapping = {}'.format(score[i], playerIdx))
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
            # 后面的是不让小鸟跌落到base图像（地面）上。
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
    """
    返回图片的alpha通道的值。即使用图片的alpha通道的值作为mask。
    如果在两个精灵重叠区域（碰撞区域）中，存在两个精灵的alpha值都不为零的位置说明发生了碰撞。
    """
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
