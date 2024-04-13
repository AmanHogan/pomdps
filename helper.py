import numpy as np
import matplotlib.pyplot as plt
from globals import *
import random


def plot_graphs(agents, t, e):
    for i in range(len(agents)):
        episodes = list(range(1, agents[i].number_of_eps + 1))
        plt.plot(episodes, agents[i].total_rewards, label=f"Trial {i + 1}")
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.title(t)
    plt.legend()
    plt.savefig(t + "_num_eps_" + str(e) + ".png")
    plt.show()
    


def generate_obstacles(num_obstacles):
    obstacles = set()
    while len(obstacles) < num_obstacles:
        x = random.randint(1, X_DIRECTION)
        y = random.randint(1, Y_DIRECTION)
        obstacles.add((x, y))
    return list(obstacles)

def generate_random_goal():
    x = random.randint(1, X_DIRECTION)
    y = random.randint(1, Y_DIRECTION)
    return x, y