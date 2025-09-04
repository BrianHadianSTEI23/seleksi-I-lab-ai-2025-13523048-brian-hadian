
import numpy as np


class Map : 
    def __init__(self) -> None:
        # init of map 
        self.map : dict[tuple[int, int], list[str]] = { # jangan lupa input semuanya
            (0, 0) : ["stench"],
        }

    def evaluateState(self, state, action, winCount: int, goldAcquired: bool):
        actFinalValue = 0
        done = False
        newState = state  

        if action == "grab":
            if state != (1, 1):
                actFinalValue -= 1
            else:
                actFinalValue += 1000
                goldAcquired = True

        elif action == "climb":
            if goldAcquired and state == (3, 0):
                actFinalValue += 1000
                done = True
                winCount += 1
            else:
                actFinalValue -= 1

        else:  # movement
            if action == "up":
                newState = (max(state[0] - 1, 0), state[1])
            elif action == "down":
                newState = (min(state[0] + 1, 3), state[1])
            elif action == "right":
                newState = (state[0], min(state[1] + 1, 3))  
            elif action == "left":
                newState = (state[0], max(state[1] - 1, 0))
            elif action == "turn_around":
                newState = state  # no movement

            # check environment
            actEnv = self.map.get(newState, [])
            for el in actEnv:
                if el in ("pit", "wumpus"):
                    actFinalValue -= 1000 + 1  # -1 for moving
                    done = True
                else:
                    actFinalValue -= 1

        return newState, actFinalValue, done, goldAcquired


    @staticmethod
    def envEquvalenceValue(env):
        if (env == "pit") : return -1000
        if (env == "wumpus") : return -1000

class QLearningWumpus : 

    def __init__(self, learningRate : float, discountFactor : float, epsilon : float) -> None:

        # main component for q learning
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.action = ["grab", "climb", "up", "down", "left", "right", "turn_around"]
        self.qTable : dict[tuple[int, int], dict[str, float]] = {
            (i, j): {act: 0.0 for act in self.action} for i in range(4) for j in range(4)
        }
        self.environment = ["pit", "wumpus", "stench", "breeze"]
        self.currState = (3, 0)
        self.epsilon = epsilon
        self.goldAcquired : bool = False
        self.win : bool = False
        self.winCount : int = 0
        pass

        
    def updateQTable(self, lastState, reward, currentState, action) :
        # this function will update the q table based on the perceived MAX val on that current state
        currMaxQTable = max(self.qTable[currentState].values())
        self.qTable[lastState][action] += self.learningRate * (reward + self.qTable[lastState][action] + (self.discountFactor * currMaxQTable) - self.qTable[lastState][action])
        return

    def train(self, map : Map) : 

        while (self.winCount < 30) : # winning count is hardcoded
            # given map and policy of epsilon-greedy (exploration v exploitation), deterimine first the value of epsilon and thus action
            if np.random.rand() < self.epsilon:
                action = self.action[np.random.randint(0, len(self.action))]
            else:
                # get the dict of move and its value in current time
                action = max(self.qTable[self.currState], key=lambda a: self.qTable[self.currState][a])

            # move and then update the Q table based on the rule
            lastState = self.currState
            newState, reward, done, self.goldAcquired = map.evaluateState(self.currState, action, self.winCount, self.goldAcquired)

            # update Q table
            self.updateQTable(lastState, reward, newState, action)

            # update agent state
            self.currState = newState

            if done and self.goldAcquired :
                self.winCount += 1

            # reset almost all the attributed and do enough iteration, it will do good enough or it will stop (divergence, pls i hope not)
            if done :
                self.currState = (3, 0)
                self.goldAcquired = False
                self.win = False

        return 
    

