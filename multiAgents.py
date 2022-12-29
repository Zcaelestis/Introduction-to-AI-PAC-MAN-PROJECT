# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        print(successorGameState)
        newPos = successorGameState.getPacmanPosition()
        print(newPos)
        newFood = successorGameState.getFood()
        print(newFood)
        newGhostStates = successorGameState.getGhostStates()
        print(newGhostStates)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print(newScaredTimes)
        "*** YOUR CODE HERE ***"
        infiniteDis=-10**10
        ateFood = currentGameState.getFood()
        fdList=ateFood.asList()
        for newState in newGhostStates:
            if newState.scaredTimer == 0 and newState.getPosition() == newPos:
                return infiniteDis
        for foods in fdList:
            disFdNewpos=-(manhattanDistance(foods,newPos))
            if disFdNewpos >= infiniteDis:
                infiniteDis=disFdNewpos
        if action == 'stop':
            return infiniteDis
        return infiniteDis


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):
            depth += 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            for a in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, a)
                v = max(v, min_value(successor, 1, depth))
            return v

        def min_value(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('inf')
            for a in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, a)
                if agentIndex == (gameState.getNumAgents() - 1):
                    v = min(v, max_value(successor, depth))
                else:
                    v = min(v, min_value(successor, depth, agentIndex+1))
            return v

        actions = gameState.getLegalActions(0)
        dict = {}
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            dict[action] = min_value(nextState, 0, 1)
        return max(dict, key=dict.get)

        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, alpha, beta, depth):
            depth += 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')

            for a in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, a)
                v = max(v, min_value(successor, alpha, beta, depth, 1))
                if (v > beta):
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(gameState, alpha, beta, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('inf')
            for a in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, a)
                if agentIndex == (gameState.getNumAgents() - 1):
                    v = min(v, max_value(successor, alpha, beta, depth))
                    if (v < alpha):
                        return v
                    beta = min(beta, v)
                else:
                    v = min(v, min_value(successor,
                                         alpha, beta, depth, agentIndex+1))
                    if (v < alpha):
                        return v
                    beta = min(beta, v)
            return v
        alpha = float('-inf')
        beta = float('inf')
        actions = gameState.getLegalActions(0)
        dict = {}
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            dict[action] = min_value(nextState, alpha, beta, 0, 1)
            v = dict[action]
            if (v > beta):
                return dict
            alpha = max(v, alpha)
        return max(dict, key=dict.get)

        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth):
            depth += 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            for a in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, a)
                v = max(v, exp_value(successor, 1, depth))
            return v

        def exp_value(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = 0
            probability = 1/len(gameState.getLegalActions(agentIndex))
            for a in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, a)
                if agentIndex == (gameState.getNumAgents() - 1):
                    v += max_value(successor, depth)*probability
                else:
                    v += exp_value(successor, depth,
                                   agentIndex+1)*probability
            return v

        actions = gameState.getLegalActions(0)
        dict = {}
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            dict[action] = exp_value(nextState, 0, 1)
        return max(dict, key=dict.get)

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: As we know if Pacman get a higher score it should keep far away
    from ghost, I will restrict the distance between Pacman and Ghosts higher than 1 or 2
    (which both work)
    and then keep Pacman seeking for food and capsules. The main tool is manhattanDistance function
        <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    minimumDis = float('inf')
    ReminimumDis = False
    currentScore = currentGameState.getScore()
    evalNum=0
    #   KEEP AWAY GHOSTS
    for newGhosts in currentGameState.getGhostPositions():
        DisGhost=util.manhattanDistance(currentGameState.getPacmanPosition(),newGhosts)
        if DisGhost < 1:
            evalNum = minimumDis
    #   SEEK FOR FOOD
    for newFood in currentGameState.getFood().asList():
        DisFd=util.manhattanDistance(currentGameState.getPacmanPosition(),newFood)
        if DisFd < minimumDis:
            minimumDis = DisFd
            ReminimumDis = True
    if ReminimumDis == True:
        evalNum = minimumDis + evalNum
    
    evalNum -= currentScore*999
    return -evalNum
# Abbreviation
better = betterEvaluationFunction
