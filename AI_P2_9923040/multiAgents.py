from Agents import Agent
from Game import GameState
import util
import random

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.index = 0 # your agent always has index 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(self.index)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', **kwargs):
        self.index = 0 # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent which extends MultiAgentSearchAgent and is supposed to be implementing a minimax tree with a certain depth.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        But before getting your hands dirty, look at these functions:

        gameState.isGameFinished() -> bool
        gameState.getNumAgents() -> int
        gameState.generateSuccessor(agentIndex, action) -> GameState
        gameState.getLegalActions(agentIndex) -> list
        self.evaluationFunction(gameState) -> float
        """
        return self.MinMax_Decision(state)[1]

    def MinMax_Decision(self, gameState, index= 0, layer= 0):
        """
        This function provides a status that determines whether 
                    the min or max function should be called.
        """
        '''
        0=0 1=1 2=0 3=1 4=0
        0=0 1=1 2=2 3=3 4=0
        => output index is the remainder is devided by past index to number of agent 
        '''
        index = index % gameState.getNumAgents() 

        if gameState.isGameFinished() or layer == self.depth or (not gameState.getLegalActions(index)):
            # return None, scoreEvaluationFunction(gameState)
            return self.evaluationFunction(gameState), None

        if index:
            return self.Min_Value(gameState, index, layer)

        else:
            layer += 1
            return self.Max_Value(gameState, index, layer)

    def Max_Value(self, gameState, index, layer):
        # initialize Value
        v = 0 # float("-inf") 
        best_act = tuple() # None
        actions = gameState.getLegalActions(index)
        # for any action in actions we have one state ...
        for action in actions:
            child = gameState.generateSuccessor(index, action)
            value = self.MinMax_Decision(child, index+1, layer)[0]
            v, best_act = max((value, action), (v, best_act))
        return v, best_act

    def Min_Value(self, gameState, index, layer):
        v = 64 # float("inf") # initialze Value
        best_act = tuple() # None
        # for any action in actions we have one state ...
        actions = gameState.getLegalActions(index)
        for action in actions:
            child = gameState.generateSuccessor(index, action)
            value = self.MinMax_Decision(child, index+1, layer)[0]
            v, best_act = min((value, action), (v, best_act))
        return v, best_act

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        You should keep track of alpha and beta in each node to be able to implement alpha-beta pruning.
        """
        alpha = float("-inf")
        betta = float("inf")
        return self.AlphaBetaPruning(gameState)[1]

    def AlphaBetaPruning(self, gameState, index= 0, layer= 0, alpha= float("-inf"), betta= float("inf")):
        index = index % gameState.getNumAgents()
        if gameState.isGameFinished() or layer == self.depth or (not gameState.getLegalActions(index)):
            # return None, scoreEvaluationFunction(gameState)
            return self.evaluationFunction(gameState), None

        if index:
            return self.Min_Value(gameState, index, layer, alpha, betta)
        else:
            layer += 1
            return self.Max_Value(gameState, index, layer, alpha, betta)

    def Max_Value(self, gameState, index, layer, alpha, betta):
        # initialize Value
        v = 0 # float("-inf") 
        best_act = tuple() # None
        actions = gameState.getLegalActions(index)
        # for any action in actions we have one satate ...
        for action in actions:
            child = gameState.generateSuccessor(index, action)
            value = self.AlphaBetaPruning(child, index+1, layer, alpha, betta)[0]
            v, best_act = max((value, action), (v, best_act))
            if v > betta:
                return v, best_act
            alpha = max(alpha, v)
        return v, best_act
    
    def Min_Value(self, gameState, index, layer, alpha, betta):
        v = 64 # float("inf") # initialze Value
        best_act = tuple() # None
        actions = gameState.getLegalActions(index)
        for action in actions:
            child = gameState.generateSuccessor(index, action)
            value = self.AlphaBetaPruning(child, index+1, layer, alpha, betta)[0]
            v, best_act = min((value, action), (v, best_act))
            if v < alpha:
                return v, best_act
            betta = min(betta, v)
        return v, best_act
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent which has a max node for your agent but every other node is a chance node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All opponents should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.Expectimax_Decision(gameState)[1]

    def Expectimax_Decision(self, gameState, index= 0, layer= 0):
        index = index % gameState.getNumAgents()
        if gameState.isGameFinished() or layer == self.depth or (not gameState.getLegalActions(index)):
            # return None, scoreEvaluationFunction(gameState)
            return self.evaluationFunction(gameState), None
            
        if index:
            return self.Exp_Value(gameState, index, layer)

        else:
            layer += 1
            return self.Max_Value(gameState, index, layer)

    def Max_Value(self, gameState, index, layer):
        # initialize Value
        v = float("-inf") 
        best_act = tuple() # None
        actions = gameState.getLegalActions(index)
        # for any action in actions we have one state ...
        for action in actions:
            child = gameState.generateSuccessor(index, action)
            value = self.Expectimax_Decision(child, index+1, layer)[0]
            v, best_act = max((value, action), (v, best_act))
        return v, best_act
    def Exp_Value(self, gameState, index, layer):
        value = 0
        actions = gameState.getLegalActions(index)
        for action in actions:
            child = gameState.generateSuccessor(index, action)
            value += self.Expectimax_Decision(child, index+1, layer)[0]
        return value/len(actions), None
    
def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function.

    You are asked to read the following paper on othello heuristics and extend it for two to four player rollit game.
    Implementing a good stability heuristic has extra points.
    Any other brilliant ideas are also accepted. Just try and be original.

    The paper: Sannidhanam, Vaishnavi, and Muthukaruppan Annamalai. "An analysis of heuristics in othello." (2015).

    Here are also some functions you will need to use:
    
    gameState.getPieces(index) -> list
    gameState.getCorners() -> 4-tuple
    gameState.getScore() -> list
    gameState.getScore(index) -> int

    """
    
    Max_P_Coins = currentGameState.getScore(0)
    Min_P_Coins = currentGameState.getScore(1)
    parity_heu = 100 * (Max_P_Coins - Min_P_Coins)/(Max_P_Coins + Min_P_Coins)
    # corners
    status_corners = currentGameState.getCorners()
    max_Agent_cor = status_corners.count(0)
    min_Agent_cor = status_corners.count(1)
    total_corners_heu = max_Agent_cor + min_Agent_cor
    total_corners_heu = 100 * (max_Agent_cor - min_Agent_cor)/total_corners_heu if total_corners_heu else 0
    # mobility
    max_Agent_mob = len(currentGameState.getLegalActions(0))
    min_Agent_mob = len(currentGameState.getLegalActions(1))
    total_mobility_heu = max_Agent_mob + min_Agent_mob
    total_mobility_heu = 100 * (max_Agent_mob - min_Agent_mob)/total_mobility_heu if total_mobility_heu else 0
    # stability
    static_grid = [
                [+4, -3, +2, +2, +2, +2, -3, +4],
                [-3, -4, -1, -1, -1, -1, -4, -3],
                [+2, -1, +1, +0, +0, +1, -1, +2],
                [+2, -1, +0, +1, +1, +0, -1, +2],
                [+2, -1, +0, +1, +1, +0, -1, +2],
                [+2, -1, +1, +0, +0, +1, -1, +2],
                [-3, -4, -1, -1, -1, -1, -4, -3],
                [+4, -3, +2, +2, +2, +2, -3, +4]]
    Max_P = currentGameState.getPieces(0)
    Max_static = 0
    for position in Max_P:
        Max_static += static_grid[position[0]][position[1]]  
    Min_P = currentGameState.getPieces(1)
    Min_static = 0
    for position in Min_P:
        Min_static += static_grid[position[0]][position[1]]  
    stability_heu = Max_static + Min_static
    stability_heu = 100 * (Max_static + Min_static)/stability_heu \
                                            if stability_heu else  0
    # return 0.64*(0.35*total_corners_heu+0.3*total_mobility_heu+0.25*stability_heu+0.1*parity_heu)
    return 0.21*total_corners_heu+0.19*total_mobility_heu+0.16*stability_heu+0.07*parity_heu

# Abbreviation
better = betterEvaluationFunction