

import argparse
import time
import numpy as np
from globals import *
from learners import (QLearningAgent, QMDPAgent, ReplicatedQLearning, LinearQLearning)
from helper import *
from logger.logger import log


def main():
    log("======== START ========")
    parser = argparse.ArgumentParser(description='A simple command-line argument parser example.')
    parser.add_argument('--start_state', '-s', type=str, help='start state of agent as a string "x,y,orientation"', default="(1, 1, 'LEFT')")
    parser.add_argument('--episode_num', '-en', type=int, help='number of episodes', default=18)
    parser.add_argument('--episode_length', '-el', type=int, help='length of episodes', default=1000)
    parser.add_argument('--alpha', type=float, help='learning rate', default=0.01)
    parser.add_argument('--gamma', type=float, help='discount factor', default=1.0)
    parser.add_argument('--epsilon', type=float, help='exploration factor', default=0.4)
    parser.add_argument('--num_obstacles', '-o', type=int, help='number of obstacles', default=5)
    parser.add_argument('--trials', '-t', type=int, help='number of trials', default=4)

    args = parser.parse_args()

    num_obstacles = args.num_obstacles
    start_state = args.start_state
    start_state_str = args.start_state.strip("()")
    x, y, orientation = start_state_str.split(',')
    start_state = (int(x), int(y), orientation.strip().strip("'"))
    episode_num = args.episode_num
    episode_length = args.episode_length
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    trials = args.trials

    agents = []
    qmdp_agents = []
    linear_agents = []
    replicated_agents = []

    for i in range(trials):
        
        obstacles = generate_obstacles(num_obstacles)
        goal = generate_random_goal()
        
        log(str(vars(args)) + "\n" + "OBSTACLES:" + str(obstacles) + "\n" + "GOAL:" + str(goal))

        log("Trial: " + str(i+1) + ". Starting Q Learning")
        agent = QLearningAgent(QTABLE, alpha, gamma, epsilon, obstacles, start_state, goal, episode_num, episode_length)
        agent.q_learn()
        agents.append(agent)
        log("Trial: " + str(i+1) + ". Finished Q Learning. Goal finding efficiency: " + str(agent.goals/episode_num))

        log("Trial: " + str(i+1) + ". Starting Linear Q Learning")
        linear_agent = LinearQLearning(QTABLE, alpha, gamma, epsilon, obstacles, start_state, goal, episode_num, episode_length)
        linear_agent.q_learn()
        linear_agents.append(linear_agent)
        log("Trial: " + str(i+1) +". Finished Linear Q Learning. Goal finding efficiency: " + str(linear_agent.goals/episode_num))

        log("Trial: " + str(i+1) + ". Starting Replicated Learning")
        replicated_agent = ReplicatedQLearning(QTABLE, alpha, gamma, epsilon, obstacles, start_state, goal, episode_num, episode_length)
        replicated_agent.q_learn()
        replicated_agents.append(replicated_agent)
        log("Trial: " + str(i+1) +". Finished Replicated Learning Q Learning. Goal finding efficiency: " + str(replicated_agent.goals/episode_num))


        log("Trial: " + str(i+1) + ". Starting QMDP Learning")
        qmdp_agent = QMDPAgent(agent.q_table, alpha, gamma, epsilon, obstacles, start_state, goal, episode_num, episode_length)
        qmdp_agent.q_learn()
        qmdp_agents.append(qmdp_agent)
        log("Trial: " + str(i+1) + ". Finished Starting QMDP Learning. Goal finding efficiency: " + str(qmdp_agent.goals/episode_num))


        # plot_graphs(agents, "Q_Learning", episode_num)
        # plot_graphs(linear_agents, "Linear_Q_Learning", episode_num)
        # plot_graphs(replicated_agents, "Replicated_Q_Learning", episode_num)
        # plot_graphs(qmdp_agents, "QMDP_Learning", episode_num)

    plot_graphs(agents, "Q_Learning", episode_num)
    plot_graphs(linear_agents, "Linear_Q_Learning", episode_num)
    plot_graphs(replicated_agents, "Replicated_Q_Learning", episode_num)
    plot_graphs(qmdp_agents, "QMDP_Learning", episode_num)


    log("======== FINISH ========")


if __name__ == "__main__":
    main()