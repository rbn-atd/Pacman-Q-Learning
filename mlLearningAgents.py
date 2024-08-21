# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        self.state=state
        
    def __eq__(self, other):
        """
        Allows two states to be compared.
        
        Compares all relevant features including, the exact state, the position of pacman,
        the position of the ghosts, position of the food and position of capsules to ensure the correct state action pair is
        updated for q values and counts.
        
        Args: 
            other: another GameStateFeatures object to compare to
            
        Returns:
            Boolean indicating if all features in the state are equal.
        
        """
        
        # store all the booleans in variables since doing them all in the return state makes
        # the line very wide and unsightly.
        stateEq = self.state == other.state
        ghostEq = self.state.getGhostPositions() == other.state.getGhostPositions()
        foodEq = self.state.getFood() == other.state.getFood() 
        pacmanEq = self.state.getPacmanPosition() == other.state.getPacmanPosition()
        capsuleEq = self.state.getCapsules() == other.state.getCapsules()
        
        return hasattr(other, 'state') and stateEq  and ghostEq and foodEq and pacmanEq and capsuleEq

    def __hash__(self):
        """
        Allows states to be keys of dictionaries.
        
        Creates the combined hash of the state's pacman position, ghost positions and food positions.
        Each feature is hashed individually (not as their list representation).
        Tries hashing actual feature, if type error is thrown then hash its string conversion instead.
        
        Returns:
            Hash of all the features including state, ghost positions, pacman position, food positions and capsule positions
        
        """
        pacmanPosition = self.state.getPacmanPosition()
        ghosts = self.state.getGhostPositions()
        food = self.state.getFood()
        capsules = self.state.getCapsules()
        
        hashValue = 0
        # Try hashing pacman position, ghost position or food features, else hash string of feature
        try:
            hashValue ^= hash(pacmanPosition)
        except TypeError:
            hashValue ^= hash(str(pacmanPosition))
        for ghost in ghosts:
            try:
                hashValue ^= hash(ghost)
            except TypeError:
                hashValue ^= hash(str(ghost))
        for food in food:
            try:
                hashValue ^= hash(food)
            except TypeError:
                hashValue ^= hash(str(food))
        for capsule in capsules:
            try:
                hashValue ^= hash(capsule)
            except TypeError:
                hashValue ^= hash(str(capsule))
            
        return hashValue


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.5,
                 epsilon: float = 0.05,
                 gamma: float = 0.9,
                 maxAttempts: int = 20,
                 numTraining: int = 20):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
            
        Also defined is:
            episodesSoFar: how many pacman games, or episodes, have been completed
            qValues: dictionary of (GameStateFeatures, Direction) key pairs representing a Q-value.
            counts: dictionary of (GameStateFeatures, Direction) key pairs representing the 
                    frequency the Direction has been taken in that GameState
            previousState: stores the state explored in the previous episode
            previousAction: stores the action taken in the previous episode
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Store Q-value of (state, action) pairs
        self.qValues = util.Counter()
        # Store frequency of (state, action) pairs
        self.counts = util.Counter()
        # Stores the previous (state, action) pair pacman was in
        self.previousState = None
        # Stores the previous action pacman has taken
        self.previousAction = []

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
            
            This is calculated by finding the difference in the built in score
            between the previous state and the current state.
        """
        # Simply gets the built in score for the game finds the difference between the previous state and the current state
        return endState.getScore() - startState.getScore()


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action) value that is stored in qValues
        
        """

        return self.qValues[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
            
            returns 0 if there are no value updates currently.
        """
        
        qVals=[]
        
        # Iterate over all legal actions and store their respective Q-values in a list
        for action in state.state.getLegalActions():
            qVal = self.getQValue(state,action)
            qVals.append(qVal)
        # If there are no legal actions, return 0 else return the maximum of the qVals list
        if len(qVals) == 0:
            return 0.0
        else:
            return max(qVals)
        

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
            
            This is calculated using the general formula for the update rule,
            current_estimate = current estimate + alpha(target - current_estimate)
            where target = reward + (gamma*maxQ(current_estimate))
        """
        
        self.updateCount(state, action)
        qMaxxing = self.maxQValue(nextState)
        target = reward + (self.gamma * qMaxxing)
        # general formula for utility updates implemented for Q-value updates
        self.qValues[(state, action)] = self.getQValue(state, action) + self.getAlpha() * (target-self.getQValue(state, action))
                
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        # Increment the (state, action) pair frequency by 1
        self.counts[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
            by finding its value in counts with the (state, action) pair key.
        """
                
        return self.counts[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
            
            This is calculate by determining if the frequency value of a state action pair is less than the
            max attempts defined. If it is less then we +5 onto the Q value (or utility) of the state action pair
            Otherwise return the utility unaffected.
    
        """
        
        if counts < self.getMaxAttempts() :
            print("BOMBOCLAAT")
            # +5 utility bonus for lesser used actions in a state, else return normal utility
            return utility+5
        else:
            return utility
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
            
            Action is determined using an epsilon-greedy approach, where it can either take a random legal action with
            epsilon probability, or a greedy action with 1-epsilon probability. The greedy action is determined by applying
            explorationFn to all legal moves which adds extra reward to Q-values that have been less used in the game.
            The values are finally maximised to find the best action. The agent then learns from the previous action and state.
            The last state and action then gets updated with the current state and action determined.
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # logging to help you understand the inputs, feel free to remove
        print("Legal moves: ", legal)
        print("Pacman position: ", state.getPacmanPosition())
        print("Ghost positions:", state.getGhostPositions())
        print("Food locations: ")
        print(state.getFood())
        print("Score: ", state.getScore())

        stateFeatures = GameStateFeatures(state)

        # Now pick what action to take.
        # The current code shows how to do that but just makes the choice randomly.
        
        # flipCoin with epsilon probability to take a random choice (exploration) or 1-epsilon for greedy choice (exploitation)
        # action selection is a lambda function which finds a legal action with the maximal Q-value after applying the exploration bonus from explorationFn
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)
        else:
            action = max(legal, key=lambda action: self.explorationFn(self.getQValue(stateFeatures, action), self.getCount(stateFeatures, action)))

        # perform a Q-learning update
        if self.previousState is not None:
                reward = self.computeReward(self.previousState.state, state)
                self.learn(self.previousState, self.previousAction[-1], reward, stateFeatures)    
        # update previous action and state
        self.previousState=stateFeatures
        self.previousAction.append(action)
        return action
    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
            
            Performs a final Q-learning update for the final state of the game so the agent can learn to favour 
            win games vs loss games.
            
            Previous action and previous state both get reset so the next episode doesn't learn from them in the initial
            game.
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        
        # final state Q-learning update to ensure the agent learns from a loss or win.
        reward = self.computeReward(self.previousState.state, state)
        self.learn(self.previousState, self.previousAction[-1], reward, GameStateFeatures(state))
        # reset the previous state and action so the next episode does not learn from the previous episode.
        self.previousState=None
        self.previousAction=[]

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
            
            

