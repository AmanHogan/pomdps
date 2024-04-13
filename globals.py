"""Contains globals that are defined in the homework. You can change these values if you want"""

TRIALS = 4
"""Number of trials per agent"""

GOAL_REWARD = 100
"""Reward for reaching goal location"""

OBSTACLE_REWARD = -10
"""Reward for hitting an obstacle"""

X_DIRECTION = 10
"""max x direction"""

Y_DIRECTION = 20
"""max y direction"""

OREINTATION = ['UP','RIGHT','LEFT','DOWN']
"""different orientations of the agent"""

START_STATE = (1, 1, 'LEFT')
"""start state"""

ACTION_SET = ['TURN RIGHT', 'TURN LEFT', 'FORWARD', 'BACKWARD']
"""different action that the agent can perform"""

STRATEGY = ['explore', 'exploit']
EXPLORE = 'explore'
EXPLOIT = 'exploit'

QTABLE = {}
"""
Q-table that keeps track of the avg rewards of each state-action value.
"""
for x in range(1,X_DIRECTION+1):
    for y in range(1, Y_DIRECTION+1):
        for o in OREINTATION:
            state = (x, y, o)
            QTABLE[state] = {}
            for action_pair in ACTION_SET:
                QTABLE[state][action_pair] = 0

HIT_NONE = 'CLEAR'
"""Denotes the agent is clear of any borders or walls"""

HIT_WALL = 'WALL'
"""Denotes the agent has hit a wall"""

HIT_OBSTACLE = 'OBSTACLE'
"""Denotes the agent has hit an obstacle"""

HIT_GOAL = 'GOAL'
"""Denotes the agent has hit the goal"""

BORDER_WALLS = []
"""List of (x,y) points that are on the border of the grid"""

ALL_CELLS = []

NON_BORDER = []
"""List of (x,y) points that are not on the border of the grid"""

CORNERS = [(1,1), (1, Y_DIRECTION), (X_DIRECTION,1), (X_DIRECTION,Y_DIRECTION)]
"""List of (x,y) points that are on the corners of the grid"""

for i in range(1, X_DIRECTION + 1):
    BORDER_WALLS.append((i, 1))
    BORDER_WALLS.append((i, Y_DIRECTION))
for j in range(1, Y_DIRECTION + 1):
    BORDER_WALLS.append((1, j))
    BORDER_WALLS.append((X_DIRECTION, j))
for i in range(1, X_DIRECTION + 1):
    for j in range(1, Y_DIRECTION + 1):
        ALL_CELLS.append((i, j))
for cell in ALL_CELLS:
    if cell not in BORDER_WALLS:
        NON_BORDER.append(cell)

STATE_SPACE = []
"""State space for the environment"""
for states in QTABLE:
    STATE_SPACE.append(QTABLE[state])