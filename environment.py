"""Models the grid world environment for fully observable environemnt."""

import sys
import numpy as np
import time
from globals import * 
import numpy as np
from logger.logger import log


class FullyObservedEnvironmentModel:
    """
    Models the grid world environment for fully observable environemnt.
    """

    def __init__(self, state, goal, obstacles):
        self.curr_state = state # snapshot of env
        self.t = 0 # time that increments by timestep
        self.goal_state = goal
        self.obstacles = obstacles
    
    def state_transition(self, state, action):
        """
        Transitions the environment to a new state given actions.

        Args:
            state (state): snapshot of environment
            action (action): action chosen by agent

        Returns:
            next_state: new snapshot of environment
        """

        x, y, orien = state

        if action == 'TURN RIGHT':
            if orien == 'UP':
                orien = 'RIGHT'
            elif orien == 'RIGHT':
                orien = 'DOWN'
            elif orien == 'DOWN':
                orien = 'LEFT'
            elif orien == 'LEFT':
                orien = 'UP'
            
        elif action == 'TURN LEFT':
            if orien == 'UP':
                orien = 'LEFT'
            elif orien == 'LEFT':
                orien = 'DOWN'
            elif orien == 'DOWN':
                orien = 'RIGHT'
            elif orien == 'RIGHT':
                orien = 'UP'
            
        elif action == 'FORWARD':
            if orien == 'UP':
                y = y + 1
            elif orien == 'DOWN':
                y = y - 1
            elif orien == 'RIGHT':
                x = x + 1
            elif orien == 'LEFT':
                x = x - 1

        elif action == 'BACKWARD':
            if orien == 'UP':
                y = y - 1
            elif orien == 'DOWN':
                y = y + 1
            elif orien == 'RIGHT':
                x = x - 1
            elif orien == 'LEFT':
                x = x + 1

        new_state = (x, y, orien)
        return new_state

    def step(self, state, action):
        """
        Gets a new state and reward based on state\n
        Args:
            state (state): snapshot of environment
            action (action): action chosen by agent
        Returns:
            new_state, reward: the new state of envirnoment and the reward for being in that state
        """

        next_state_ = self.state_transition(state, action)
        next_reward, next_state = self.reward_func(state, next_state_)
        self.t = self.t + 1

        return next_state, next_reward
    
    def print_state(self, state, action, next_state, reward, ep, it):
        print('-------------------------')
        print('EPISODE:', ep, 'ITERATION', it )
        print("STATE:", state, " ACTION: ",action)
        print("NEXT STATE: ", next_state, " REWARD: ", reward)
        print('-------------------------')

    def reward_func(self, state, next_state):
        """
        Returns the reward for being in the new state, and also performs state validation

        Args:
            state (tuple): prev state
            next_state (tuple): next state

        Returns:
            float, tuple: reward, validated state
        """
        
        x_1, y_1, orien_1 = state
        x_2, y_2, orien_2 = next_state

        reward = 0
        
        if (x_2, y_2)in self.obstacles:
            reward = OBSTACLE_REWARD
            x_2 = x_1
            y_2 = y_1
            orien_2 = orien_1

        if (x_2,y_2) == self.goal_state:
            reward = GOAL_REWARD

        if x_2 > X_DIRECTION or y_2 > Y_DIRECTION or x_2 < 1 or  y_2 < 1:
            x_2 = x_1
            y_2 = y_1
            orien_2 = orien_1
            reward = OBSTACLE_REWARD
           
        next_state = (x_2, y_2, orien_2)
        return reward, next_state

class PartialObservedEnvironmentModel:
    """
    Models the grid world environment for a partially observed environment
    """

    def __init__(self, state, goal, obstacles):
        self.state = state # tru current state accesible only to environemnt
        self.t = 0 # time that increments by timestep
        self.prev_state = state
        self.goal_state = goal
        self.obstacles = obstacles
    
    def state_transition(self, state, action):
        """
        Transitions the environment to a new state given actions.

        Args:
            state (state): snapshot of environment
            action (action): action chosen by agent

        Returns:
            next_state: new snapshot of environment
        """

        x, y, orien = state

        if action == 'TURN RIGHT':
            if orien == 'UP':
                orien = 'RIGHT'
            elif orien == 'RIGHT':
                orien = 'DOWN'
            elif orien == 'DOWN':
                orien = 'LEFT'
            elif orien == 'LEFT':
                orien = 'UP'
            
        elif action == 'TURN LEFT':
            if orien == 'UP':
                orien = 'LEFT'
            elif orien == 'LEFT':
                orien = 'DOWN'
            elif orien == 'DOWN':
                orien = 'RIGHT'
            elif orien == 'RIGHT':
                orien = 'UP'
            
        elif action == 'FORWARD':
            if orien == 'UP':
                y = y + 1
            elif orien == 'DOWN':
                y = y - 1
            elif orien == 'RIGHT':
                x = x + 1
            elif orien == 'LEFT':
                x = x - 1

        elif action == 'BACKWARD':
            if orien == 'UP':
                y = y - 1
            elif orien == 'DOWN':
                y = y + 1
            elif orien == 'RIGHT':
                x = x - 1
            elif orien == 'LEFT':
                x = x + 1

        new_state = (x, y, orien)
        return new_state

    def step(self, action):
        """
        Gets a new state and reward based on state
        Args:
            state (state): snapshot of environment
            action (action): action chosen by agent
        Returns:
            new_state, reward: the new state of envirnoment and the reward for being in that state
        """

        next_state_ = self.state_transition(self.state, action) # get next state
        observed_state = self.observation_function(next_state_) # get agents point of view
        next_reward, next_state = self.reward_func(self.state, next_state_) # get reward, and validate next state
        self.t = self.t + 1 # increment time

        self.prev_state = self.state
        self.state = next_state
        print("ACTUAL STATE: ", self.state )

        return observed_state, next_reward
    
    def print_state(self, state, action, next_state, reward, ep, it):
        print('-------------------------')
        print('EPISODE:', ep, 'ITERATION', it )
        print("STATE:", state, " ACTION: ",action)
        print("NEXT STATE: ", next_state, " REWARD: ", reward)
        print('-------------------------')

    def reward_func(self, state, next_state):
        """
        Returns the reward for being in the new state, and also performs state validation

        Args:
            state (tuple): prev state
            next_state (tuple): next state

        Returns:
            float, tuple: reward, validated state
        """
        
        x_1, y_1, orien_1 = state
        x_2, y_2, orien_2 = next_state

        reward = 0
        
        # next state is in an obstacle
        if (x_2, y_2)in self.obstacles:
            reward = OBSTACLE_REWARD
            x_2 = x_1
            y_2 = y_1
            orien_2 = orien_1

        # next state is in goal
        if (x_2,y_2) == self.goal_state:
            reward = GOAL_REWARD

        # next state exceeded bounds
        if x_2 > X_DIRECTION or y_2 > Y_DIRECTION or x_2 < 1 or  y_2 < 1:
            x_2 = x_1
            y_2 = y_1
            orien_2 = orien_1
            reward = OBSTACLE_REWARD
           
        next_state = (x_2, y_2, orien_2)
        return reward, next_state
 
    def observation_function(self, next_state):
        """
        Return the observed state of the agent from the agents perspective.
        Args: next_state (tuple): true state that only the environment knows
        Returns: (tuple): observed_state
        """

        x_2, y_2, orien_2 = next_state
        observed_state = None

        # Return observation from agent's view, given the actual state
        if (x_2, y_2)in self.obstacles:
            observed_state = HIT_OBSTACLE

        elif (x_2,y_2) == self.goal_state:
            observed_state = HIT_GOAL
    
        elif x_2 > X_DIRECTION or y_2 > Y_DIRECTION or x_2 < 1 or  y_2 < 1:
            observed_state = HIT_WALL

        else:
            observed_state = HIT_NONE
           
        return observed_state

    def observation_probs(self, state, observation, action):
        """
        Determine the probability of the agent making an observation given the actual state and action.
        Since it is a deterministic environment, we assume the chance is 100%.
        
        Args:
            state (tuple): The true state of the environment.
            observation (str): The observation the agent has perceived.
            action (str): Action taken, but not used since its deterministic.
        
        Returns:
            float: Probability of making the observation given the state and action.
        """

        chance_correct = 1
        chance_wrong = 1 - chance_correct

        # What the agent should observe based on the actual state
        expected = self.observation_function(state)

        # Probability of observation
        if observation == expected:
            return chance_correct
        else:
            return chance_wrong

if __name__ == '__main__':
    start_state = (10,20,'UP')
    action = 'FORWARD'
    goal = (10,10)
    obstacles = [(5,5)]
    env = FullyObservedEnvironmentModel(start_state,goal, obstacles )
    next_state = env.state_transition(start_state, action)
    reward, next_state = env.reward_func(start_state, next_state)
    print(start_state, action)
    print(next_state)
    print(reward)
        
   