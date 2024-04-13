# Grid World Learning Experiment

This Python script conducts an experiment to evaluate different reinforcement learning algorithms in a grid world environment. It allows for the comparison of Q-learning, Linear Q-learning, Replicated Q-learning, and QMDP Learning algorithms.

## Installation

Ensure you have Python installed, along with the necessary dependencies specified in the project's requirements file.


### Running the Experiment

To run the experiment, execute the script `agents.py` with appropriate command-line arguments:

```
python -m agents --start_state "(1, 1, 'LEFT')" --episode_num 18 --episode_length 1000 --alpha 0.01 --gamma 1.0 --epsilon 0.4 --num_obstacles 5 --trials 4
```

#### Command-line Arguments

- `--start_state, -s`: Start state of the agent in the format "x,y,orientation". Default is "(1, 1, 'LEFT')".
- `--episode_num, -en`: Number of episodes. Default is 18.
- `--episode_length, -el`: Length of episodes. Default is 1000.
- `--alpha`: Learning rate. Default is 0.01.
- `--gamma`: Discount factor. Default is 1.0.
- `--epsilon`: Exploration factor. Default is 0.4.
- `--num_obstacles, -o`: Number of obstacles in the grid world. Default is 5.
- `--trials, -t`: Number of trials for each learning algorithm. Default is 4.

### Learning Algorithms

The script evaluates the following learning algorithms:

- **Q-Learning**: A basic Q-learning algorithm for fully observable environments.
- **Linear Q-Learning**: Q-learning with linear function approximation for partially observable environments.
- **Replicated Q-Learning**: A variant of Q-learning for partially observable environments.
- **QMDP Learning**: Q-learning with Markov Decision Processes for partially observable environments.


### Q-Learning Agent

The `QLearningAgent` class implements Q-learning for fully observable environments. It learns the optimal action-value function Q(s, a) by interacting with the environment and updating Q-values based on observed rewards.

### QMDP Agent

The `QMDPAgent` class implements QMDP learning for partially observable environments. It learns an approximate Q-value function Q(b, a) by maintaining a belief state over possible states and actions.

### Replicated Q-Learning

The `ReplicatedQLearning` class implements Replicated Q-learning for partially observable environments. It extends Q-learning to handle partial observability by maintaining a belief state and updating Q-values accordingly.


### Linear Q-Learning

The `LinearQLearning` class implements Linear Q-learning for partially observable environments. It approximates Q-values using a linear combination of feature weights and state features.


### Fully Observed Environment Model

The `FullyObservedEnvironmentModel` class represents a fully observable grid world environment. In this environment, the agent has complete knowledge of the entire state space at each timestep.

### Partially Observed Environment Model

The `PartialObservedEnvironmentModel` class represents a partially observable grid world environment. In this environment, the agent's observations are limited, providing only partial information about the true state.

## Contributions
- https://people.cs.umass.edu/~barto/courses/cs687/Cassandra-etal-POMDP.pdf
- https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process
- Manfred Huber
- Aman Hogan-Bailey
- The University of Texas at Arlington