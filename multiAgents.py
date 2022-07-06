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
import random, util

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
        # print("current = ", gameState.getScore())
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        from math import inf

        # get useful parameters from the current game state
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        # check if it's a win or lose state then return immediately with min or max score
        if successorGameState.isLose():
            return -10000
        elif successorGameState.isWin():
            return 10000

        food_list = newFood.asList()
        # get the distance to the closest food
        min_food = min(map(lambda x: manhattanDistance(x, newPos), food_list))
        # get the distance to the closest active ghost
        min_active_ghost = min(
            map(lambda x: inf if x.scaredTimer else manhattanDistance(x.configuration.pos, newPos),
                newGhostStates))
        # get the distance to the closest scared ghost
        min_scared_ghost = min(
            map(lambda x: inf if not x.scaredTimer else manhattanDistance(x.configuration.pos, newPos),
                newGhostStates))

        # get a list of capsules positions
        capsules = successorGameState.getCapsules()
        if capsules:  # if list is not empty get distance to the closest capsule
            min_capsule = min(map(lambda x: manhattanDistance(x, newPos), capsules))
        else:  # otherwise => no capsules so set distance to closest capsule to inf
            min_capsule = inf

        # add a wait penalty if current action is to stop to prevent pacman from lying still
        wait_penalty = 0
        if action == 'Stop':
            wait_penalty = -50

        # final evaluation of state is a weighted combination of the many parameters
        # 1- add current state score because it encourages pacman to eat adjacent food and escape adjacent ghosts
        # 2- subtract reciprocal of distance to closest active ghost to encourage pacman to get away from ghosts
        # 3- add reciprocal of distance to closest scared ghost to encourage pacman to get attack scared ghosts
        # 4- add reciprocal of distance to closest food to encourage pacman to get close to food
        # 5- add reciprocal of total food on map to encourage pacman to eat food
        # 6- add reciprocal of distance to closest capsule to encourage pacman to get close to capsules
        # 7- subtract total number capsules on map to encourage pacman to eat capsules
        return successorGameState.getScore() + \
               -2.5 / min_active_ghost + \
               4 / min_scared_ghost + \
               2 / min_food + \
               1 / len(food_list) + \
               4 / min_capsule + \
               -5 * len(capsules) + \
               wait_penalty


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
        from math import inf

        def minimax_value(state, depth):
            """
            get the best score max can get from the current state and current depth
            """

            # get the number of agents in this game
            num_agent = state.getNumAgents()
            # terminating state = win or loss or maximum depth achieved(depth is assumed to increase by 1 when all
            # agents execute 1 move)
            if depth >= self.depth * num_agent or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if depth % num_agent == 0:  # Pacman's turn
                maxi = -inf
                for action in (state.getLegalActions(0)):
                    # call the function again after pacman moves and increase depth by 1
                    maxi = max(maxi, minimax_value(state.generateSuccessor(0, action), depth + 1))
                return maxi
            else:  # Ghost's turn
                # get the index of the current ghost
                ghost = depth % num_agent
                mini = inf
                for action in (state.getLegalActions(ghost)):
                    # call the function again after current ghost moves and increase depth by 1
                    mini = min(mini, minimax_value(state.generateSuccessor(ghost, action), depth + 1))
                return mini

        # init an empty list
        action_score_pair = []
        # loop in all legal actions for Pacman
        for action in gameState.getLegalActions(0):
            # pass each action to minimax_value to get its score and append the action and its score to the list
            action_score_pair.append(
                (action, minimax_value(gameState.generateSuccessor(0, action), 1)))
        # return the action with the maximum score
        return max(action_score_pair, key=lambda x: x[1])[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        from math import inf

        # init an empty list of scores of initial actions
        init_action_scores = []

        def alpha_beta(state, depth, alpha=-inf, beta=inf):
            """
            get the best score max can get from the current state and current depth
            """

            # get the number of agents in this game
            num_agent = state.getNumAgents()
            # terminating state = win or loss or maximum depth achieved(depth is assumed to increase by 1 when all
            # agents execute 1 move)
            if depth >= self.depth * num_agent or state.isWin() or state.isLose():
                # return self.evaluationFunction(state)
                return betterEvaluationFunction(state)

            if depth % num_agent == 0:  # Pacman's turn
                maxi = -inf
                for action in (state.getLegalActions(0)):
                    # call the function again after pacman moves and increase depth by 1
                    maxi = max(maxi, alpha_beta(state.generateSuccessor(0, action), depth + 1, alpha, beta))
                    if depth == 0:  # if depth is zero add this score to the initial action scores list
                        init_action_scores.append(maxi)
                    if maxi > beta: return maxi  # pruning condition
                    # update alpha
                    alpha = max(alpha, maxi)
                return maxi
            else:  # Ghost's turn
                # get the index of the current ghost
                ghost = depth % num_agent
                mini = inf
                for action in (state.getLegalActions(ghost)):
                    # call the function again after current ghost moves and increase depth by 1
                    mini = min(mini, alpha_beta(state.generateSuccessor(ghost, action), depth + 1, alpha, beta))
                    if mini < alpha: return mini  # pruning condition
                    # update beta
                    beta = min(beta, mini)
                return mini

        # run alpha_beta on initial state, now the initial_action_scores list will be populated
        alpha_beta(gameState, 0)
        # get all legal actions for pacman for the initial game state
        pacman_actions = gameState.getLegalActions(0)
        # get the best action for pacman by getting the index of the max score in init_action_scores list
        best_action = pacman_actions[init_action_scores.index(max(init_action_scores))]
        return best_action


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
        from math import inf

        def expctimax_value(state, depth):
            """
            get the best score max can get from the current state and current depth
            """

            # get the number of agents in this game
            num_agent = state.getNumAgents()
            # terminating state = win or loss or maximum depth achieved(depth is assumed to increase by 1 when all
            # agents execute 1 move)
            if depth >= self.depth * num_agent or state.isWin() or state.isLose():
                # return self.evaluationFunction(state)
                return betterEvaluationFunction(state)

            if depth % num_agent == 0:  # Pacman's turn
                maxi = -inf
                for action in state.getLegalActions(0):
                    # call the function again after pacman moves and increase depth by 1
                    maxi = max(maxi, expctimax_value(state.generateSuccessor(0, action), depth + 1))
                return maxi
            else:  # Ghost's turn
                # get the index of the current ghost
                ghost = depth % num_agent
                mini = inf
                next_ghost_actions = state.getLegalActions(ghost)
                action_prob = 1 / len(next_ghost_actions)
                expected_score = 0
                for action in next_ghost_actions:
                    # get the expected score of all actions
                    expected_score += action_prob * min(mini, expctimax_value(state.generateSuccessor(ghost, action),
                                                                              depth + 1))
                return expected_score

        # init an empty list
        action_score_pair = []
        # loop in all legal actions for Pacman
        for action in gameState.getLegalActions(0):
            # pass each action to minimax_value to get its score and append the action and its score to the list
            action_score_pair.append(
                (action, expctimax_value(gameState.generateSuccessor(0, action), 1)))
        # return the action with the maximum score
        return max(action_score_pair, key=lambda x: x[1])[0]

    def expected_value(self, game_state, index, depth):
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = game_state.getLegalActions(index)
        expected_value = 0
        expected_action = ""

        # Find the current successor's probability using a uniform distribution
        successor_probability = 1.0 / len(legalMoves)

        for action in legalMoves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            # Calculate the action-score for the current successor
            current_action, current_value = self.get_value(successor, successor_index, successor_depth)

            # Update expected_value with the current_value and successor_probability
            expected_value += successor_probability * current_value

        return expected_action, expected_value


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    from math import inf

    # get useful parameters from the current game state
    pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()

    # check if it's a win or lose state then return immediately with min or max score
    if currentGameState.isLose():
        return -10000
    elif currentGameState.isWin():
        return 10000

    # get the distance to the closest food
    min_food = min(map(lambda x: manhattanDistance(x, pos), food_list))
    # get the distance to the closest active ghost
    min_active_ghost = min(
        map(lambda x: inf if x.scaredTimer else manhattanDistance(x.configuration.pos, pos),
            ghost_states))
    # get the distance to the closest scared ghost
    min_scared_ghost = min(
        map(lambda x: inf if not x.scaredTimer else manhattanDistance(x.configuration.pos, pos),
            ghost_states))
    # get a list of capsules positions
    capsules = currentGameState.getCapsules()
    if capsules:  # if list is not empty get distance to the closest capsule
        min_capsule = min(map(lambda x: manhattanDistance(x, pos), capsules))
    else:  # otherwise => no capsules so set distance to closest capsule to inf
        min_capsule = inf

    # final evaluation of state is a weighted combination of the many parameters
    # 1- add current state score because it encourages pacman to eat adjacent food and escape adjacent ghosts
    # 2- subtract reciprocal of distance to closest active ghost to encourage pacman to get away from ghosts
    # 3- add reciprocal of distance to closest scared ghost to encourage pacman to get attack scared ghosts
    # 4- add reciprocal of distance to closest food to encourage pacman to get close to food
    # 5- add reciprocal of total food on map to encourage pacman to eat food
    # 6- add reciprocal of distance to closest capsule to encourage pacman to get close to capsules
    # 7- subtract total number capsules on map to encourage pacman to eat capsules
    return currentGameState.getScore() + \
           -2.5 / min_active_ghost + \
           4 / min_scared_ghost + \
           2 / min_food + \
           1 / len(food_list) + \
           4 / min_capsule + \
           -5 * len(capsules)


def betterEvaluationFunction2(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    from math import inf

    # get useful parameters from the current game state
    pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()

    # check if it's a win or lose state then return immediately with min or max score
    if currentGameState.isLose():
        return -10000
    elif currentGameState.isWin():
        return 10000

    # get the distance to the closest food
    min_food = min(map(lambda x: manhattanDistance(x, pos), food_list))
    # get the distance to the closest active ghost
    min_active_ghost = min(
        map(lambda x: inf if x.scaredTimer else manhattanDistance(x.configuration.pos, pos),
            ghost_states))
    # get the distance to the closest scared ghost
    min_scared_ghost = min(
        map(lambda x: inf if not x.scaredTimer else manhattanDistance(x.configuration.pos, pos),
            ghost_states))
    # get a list of capsules positions
    capsules = currentGameState.getCapsules()
    if capsules:  # if list is not empty get distance to the closest capsule
        min_capsule = min(map(lambda x: manhattanDistance(x, pos), capsules))
    else:  # otherwise => no capsules so set distance to closest capsule to inf
        min_capsule = inf

    # final evaluation of state is a weighted combination of the many parameters
    # 1- add current state score because it encourages pacman to eat adjacent food and escape adjacent ghosts
    # 2- subtract reciprocal of distance to closest active ghost to encourage pacman to get away from ghosts
    # 3- add reciprocal of distance to closest scared ghost to encourage pacman to get attack scared ghosts
    # 4- add reciprocal of distance to closest food to encourage pacman to get close to food
    # 5- add reciprocal of total food on map to encourage pacman to eat food
    # 6- add reciprocal of distance to closest capsule to encourage pacman to get close to capsules
    # 7- subtract total number capsules on map to encourage pacman to eat capsules
    return currentGameState.getScore() + \
           -2.5 / min_active_ghost + \
           4 / min_scared_ghost + \
           2 / min_food + \
           1 / len(food_list) + \
           4 / min_capsule + \
           -5 * len(capsules)


# Abbreviation
better = betterEvaluationFunction
