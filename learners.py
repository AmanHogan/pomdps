""" Module contains various learners"""

import numpy as np
from environment import FullyObservedEnvironmentModel, PartialObservedEnvironmentModel
from globals import *
import copy
from logger.logger import log

class QLearningAgent:
    """
    Agent implments Q learning for fully observable environment
    """

    def __init__(self, qtable, alpha, gamma, epsilon, obstacles, start_state, goal, episode_num, episode_length):
        """
        init q learner

        Args:
            qtable (dict): q table
            alpha (float): learning rate
            gamma (float): discount factor
            epsilon (flaot): exploration rate
            obstacles (list[tuple]): list of obstacle locations
            start_state (tuple): start state
            goal (tuple): goal state
            episode_num (int): total number of episodes
            episode_length (int): max number of iterations in an episode
        """

        self.q_table = copy.deepcopy(qtable)
        self.alpha = alpha
        self.gamma = gamma
        self.explore = epsilon
        self.exploit = 1 - self.explore
        self.number_of_eps = episode_num 
        self.start_state = start_state
        self.env = FullyObservedEnvironmentModel(self.start_state, goal, obstacles)
        self.episodes = []
        self.iterations = []
        self.total_rewards = []
        self.episode_length = episode_length
        self.goal = goal
        self.goals = 0
        
    def policy(self, state):
        """
        Chooses action based on epsilon-greedy method.
            Args: state (state): snapshot of environment
            Returns: action: action to take, either random or best action.
        """

        strategy = np.random.choice(STRATEGY, 1, p=[self.explore, self.exploit])

        if strategy == EXPLORE:
            return np.random.choice(ACTION_SET)
    
        else:
            max_actions = max(self.q_table[state].items(), key=lambda x: x[1])[0]
            return max_actions

    def q_learn(self):
        """
        Performs q learning to find optimal pathing in grid
        """

        # 1. For each episode ... 
        for episode in range(self.number_of_eps):
            
            # 2. Initialize start state s
            self.episodes.append(episode)
            state = self.start_state
            total_rewards_in_episode = 0

            # 3. For reach iteration ...
            for iteration in range(self.episode_length):

                # 4. Choose action a from s using policy
                action = self.policy(state)

                # 5. Take action a and observe reward and s'
                next_state, reward = self.env.step(state, action)
            
                # 6. Update rule: Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a'(Q(s',a') - Q(s,a))]
                self.q_table[state][action] = self.q_table[state][action] + self.alpha * (reward + self.gamma*max(self.q_table[next_state].values()) - self.q_table[state][action])
                self.env.print_state(state, action, next_state, reward, episode, iteration)
                
                # 7. s = s'
                state = next_state
                total_rewards_in_episode += reward
                
                # 8. Check if reached terminating state
                x, y, _ = state
                if (x,y) == self.goal:
                    print('Reached goal')
                    self.goals += 1
                    break

            self.total_rewards.append(total_rewards_in_episode)

class QMDPAgent:
    """
    Agent implements QMDP learning for grid using the learned paramters from the q learner q(s,a)
    """

    def __init__(self, qtable, alpha, gamma, epsilon, obstacles, start_state, goal, episode_num, episode_length):
        """
        init qmdp learner

        Args:
            qtable (dict): q table
            alpha (float): learning rate
            gamma (float): discount factor
            epsilon (flaot): exploration rate
            obstacles (list[tuple]): list of obstacle locations
            start_state (tuple): start state
            goal (tuple): goal state
            episode_num (int): total number of episodes
            episode_length (int): max number of iterations in an episode
        """

        self.q_table = copy.deepcopy(qtable)
        self.alpha = alpha
        self.gamma = gamma
        self.explore = epsilon
        self.exploit = 1 - self.explore
        self.number_of_eps = episode_num 
        self.start_state = start_state
        self.env = PartialObservedEnvironmentModel(self.start_state, goal, obstacles)
        self.goals = 0
        self.episodes = []
        self.iterations = []
        self.total_rewards = []
        self.episode_length = episode_length
        self.obstacles = obstacles
        self.goal = goal

            
    def q_learn(self):
        """performs QMDP learning"""

        # 1. For each episode ...
        for episode in range(self.number_of_eps):
            
            self.episodes.append(episode)
            total_rewards_in_episode = 0

            # Init a, s, and b
            self.b = init_belief(self.start_state)
            state = self.start_state 
            action = np.random.choice(ACTION_SET)
            action_next = np.random.choice(ACTION_SET)

            # 2. For each iteration or until terminal ...
            for iteration in range(self.episode_length):

                # 3. Compute Q(b,a) using current beleif and action
                Q_b_a = compute_Q_b_a(action, self)

                # 4. Take action and get observation
                observation, reward = self.env.step(action)

                # 5. Update belief state based on observation
                update_belief(observation, self)

                # 6. Compute Q(b',a') for new beleif and actions
                Q_b_a_next_vector = {a: compute_Q_b_a(a, self) for a in ACTION_SET}

                # 7. Choose a' from b' using policy derived from Q (Epsilon-greedy)
                if np.random.random() < self.explore:
                    action_next = np.random.choice(list(ACTION_SET))
                else:
                    action_next = max(Q_b_a_next_vector, key=Q_b_a_next_vector.get)

                # 8. Update Rule for: Q learning on belief states + qmdp linear assumption
                for s in self.b:
                    if self.b[s] > 0:
                        self.q_table[state][action] += self.alpha * self.b[s] * (reward + self.gamma * Q_b_a_next_vector[action_next] - Q_b_a) 

                self.env.print_state(observation, action, observation, reward, episode, iteration)
                print("Q(b, a) = Σ b(s) * q(s,a) = ", Q_b_a)
                print("Q(b',a') = Σ b'(s') * q(s',a') =", Q_b_a_next_vector)

                # update tr
                # a = a'
                total_rewards_in_episode += reward
                action = action_next
                
                # 9. Check terminal state
                x, y, _ = self.env.state
                if (x, y) == self.goal:
                    print('Reached goal')
                    self.goals += 1
                    break

            self.total_rewards.append(total_rewards_in_episode)

class ReplicatedQLearning:
    """
    Agent implements Replicated Q learning for grid using the learned paramters from the q learner q(s,a)
    """

    def __init__(self, qtable, alpha, gamma, epsilon, obstacles, start_state, goal, episode_num, episode_length):
        """
        init replicated q learner

        Args:
            qtable (dict): q table
            alpha (float): learning rate
            gamma (float): discount factor
            epsilon (flaot): exploration rate
            obstacles (list[tuple]): list of obstacle locations
            start_state (tuple): start state
            goal (tuple): goal state
            episode_num (int): total number of episodes
            episode_length (int): max number of iterations in an episode
        """
        self.q_table = copy.deepcopy(qtable)
        self.alpha = alpha
        self.gamma = gamma
        self.explore = epsilon
        self.exploit = 1 - self.explore
        self.number_of_eps = episode_num 
        self.start_state = start_state
        self.env = PartialObservedEnvironmentModel(self.start_state, goal, obstacles)
        self.goals = 0
        self.episodes = []
        self.iterations = []
        self.total_rewards = []
        self.episode_length = episode_length
        self.obstacles = obstacles
        self.goal = goal

    def q_learn(self):
        """performs replicated q learning"""

         # 1. For each episode ...
        for episode in range(self.number_of_eps):
            
            self.episodes.append(episode)
            total_rewards_in_episode = 0

            # Init a, s, and b
            self.b = init_belief(self.start_state)
            state = self.start_state
            action = np.random.choice(ACTION_SET)
            action_next= np.random.choice(ACTION_SET)

            # 2. For each iteration or until terminal ...
            for iteration in range(self.episode_length):

                # 3. Take action and get observation
                observed_state, reward = self.env.step(action)

                # 4. Update belief state based on observation
                update_belief(observed_state, self)

                # 5. Init and Compute Q(b',a') by using new b
                Q_b_a_next_vector = {a: compute_Q_b_a(a, self) for a in ACTION_SET}

                # 6. Choose a' from b' using policy derived from Q (Epsilon-greedy)
                if np.random.random() < self.explore:
                    action_next = np.random.choice(list(ACTION_SET))
                else:
                    action_next = max(Q_b_a_next_vector, key=Q_b_a_next_vector.get)
                

                # 7. Replicated Q learning Update rule: q_a(s) = alpha * b(s) * (r + gamma * max(Q_a'(b')) - q_a(s))
                for s in self.b:
                    if self.b[s] > 0: 
                        self.q_table[s][action] += self.alpha * self.b[s] * (reward + self.gamma * Q_b_a_next_vector[action_next] - self.q_table[s][action])

                self.env.print_state(observed_state, action, observed_state, reward, episode, iteration)
                print("Q(b',a') = Σ b'(s') * q(s',a') =", Q_b_a_next_vector)

                # update tr
                # a = a'
                total_rewards_in_episode += reward
                action = action_next
                
                # 9. Check terminal state
                x, y, _ = self.env.state
                if (x, y) == self.goal:
                    print('Reached goal')
                    self.goals += 1
                    break

            self.total_rewards.append(total_rewards_in_episode)

class LinearQLearning:
    """
    Agent implementing Linear Q Learning for gridusing the learned paramters from the q learner q(s,a)
    """

    def __init__(self, qtable, alpha, gamma, epsilon, obstacles, start_state, goal, episode_num, episode_length):
        """
        Initializes agent object with the necessary information.

        Args:
            qtable (dict): Q-table of values for state-action pairs.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
        """
        self.q_table = copy.deepcopy(qtable)
        self.alpha = alpha
        self.gamma = gamma
        self.explore = epsilon
        self.exploit = 1 - self.explore
        self.number_of_eps = episode_num 
        self.start_state = start_state
        self.env = PartialObservedEnvironmentModel(self.start_state, goal, obstacles)
        self.goals = 0
        self.episodes = []
        self.iterations = []
        self.total_rewards = []
        self.episode_length = episode_length
        self.obstacles = obstacles
        self.goal = goal
        self.weights = copy.deepcopy(qtable)

    def q_learn(self):
        """performs linear q learning"""

        # 1. For each episode ...
        for episode in range(self.number_of_eps):
            
        
            self.episodes.append(episode)
            total_rewards_in_episode = 0

            # Init a, s, and b
            self.b = init_belief(self.start_state)
            state = self.start_state 
            action = np.random.choice(ACTION_SET)
            action_next = np.random.choice(ACTION_SET)
            
            # 2. For each iteration or until terminal ...
            for iteration in range(self.episode_length):

                # 3. Compute Q(b,a)
                Q_b_a = compute_Q_b_a(action, self)

                # 4. Take action and get observation
                observation, reward = self.env.step(action)  

                # 5. Update belief state based on observation
                update_belief(observation, self)

                # 6. Init and Compute Q(b',a') by using new b
                Q_b_a_next_vector = {a: compute_Q_b_a(a, self) for a in ACTION_SET}

                # 7. Choose a' from b' using policy derived from Q (Epsilon-greedy)
                if np.random.random() < self.explore:
                    action_next = np.random.choice(list(ACTION_SET))
                else:
                    action_next = max(Q_b_a_next_vector, key=Q_b_a_next_vector.get)
                
                # 8. Update rule for Linear q learning:
                # q(s,a) = q(s,a) + alpha * b(s) * (r + gamma * max(Q(b',a')) - Q(b,a))
                for s in self.b:
                    if self.b[s] > 0: 
                        self.weights[s][action] += self.alpha * self.b[s] * (reward + self.gamma * Q_b_a_next_vector[action_next] - Q_b_a)

                self.env.print_state(observation, action, observation, reward, episode, iteration)
                print("Q(b, a) = Σ b(s) * q(s,a) = ", Q_b_a)
                print("Q(b',a') = Σ b'(s') * q(s',a') =", Q_b_a_next_vector)

                # update tr
                # a = a'
                total_rewards_in_episode += reward
                action = action_next
                
                # 9. Check terminal state
                x, y, _ = self.env.state
                if (x, y) == self.goal:
                    print('Reached goal')
                    self.goals += 1
                    break

            self.total_rewards.append(total_rewards_in_episode)

def init_belief(start_state):
    """
    Initializes belief state. 
    For all intents and purposes, assumes we know for centrainty the start state. 
    This can be changed though.

    Returns:
        dict: belief state
    """

    # Initialize belief state with probabilities set to 0
    b = {}
    for x in range(1,X_DIRECTION+1,1):
        for y in range(1,Y_DIRECTION+1,1):
            for o in OREINTATION:
                state = (x, y, o)
                b[state] = 0

    b[start_state] = 1
    return b

def update_belief(observation, agent):
        """
        Updates the belief state based on the observed action and next state.
        Args: next_state (tuple): Next state observed after taking the action.
        """
        
        new_belief = {}
        
        # Increase belief for states that are adjacent to obstacles
        if observation == HIT_OBSTACLE:
            for state in agent.b:
                x, y, _ = state
                if (x+1, y) in agent.obstacles or (x-1, y) in agent.obstacles or \
                (x, y+1) in agent.obstacles or (x, y-1) in agent.obstacles:
                    likelihood = 2
                else:
                    likelihood = 0

                new_belief[state] = agent.b[state] * likelihood
        
        # Increase belief for border states
        elif observation == HIT_WALL:
            for state in agent.b:
                if state in BORDER_WALLS:
                    likelihood = 2  # Some high likelihood value
                else:
                    likelihood = 0  # No likelihood if not a border wall
                new_belief[state] = agent.b[state] * likelihood
        
        # Increase belief for non-border (interior) states
        elif observation == HIT_NONE:
            for state in agent.b:
                if state in NON_BORDER:
                    likelihood = 2
                else:
                    likelihood = 0
                new_belief[state] = agent.b[state] * likelihood
        
        # Normalize the belief state
        total_probability = sum(new_belief.values())
        if total_probability > 0:
            agent.b = {state: probability / total_probability for state, probability in new_belief.items()}
        else:
            agent.b = {state: 1.0 / len(agent.b) for state in agent.b}

def compute_Q_b_a(action, agent):
    """
    Compute the approximate Q value for a POMDP using the linear assumption.
    Q(b, a) = Σ(q(s, a)b(s))

    Args:
        action (tuple): action taken

    Returns:
        float: Q(b,a) value
    """

    Q_b_a = sum(agent.q_table[state][action] * agent.b[state] for state in agent.b)
    return Q_b_a


