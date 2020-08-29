# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 0.Import packages

# %%
import sys
import os
if sys.version_info.major == 2:
    import TKinter as tk
else:
    import tkinter as tk
import numpy as np
import pandas as pd
import time
# from IPython.display import clear_output


def clear_screen():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')
# %% [markdown]
# # 1.Creating Universe and its UI

# %%
# World constants:
UNIVERSE = [
    ['S', '0', '0', '0', '0'],
    ['W', 'W', 'W', 'W', '0'],
    ['0', '0', '0', 'W', '0'],
    ['0', 'W', '0', '0', '0'],
    ['0', 'W', 'W', 'W', 'W'],
    ['0', '0', '0', '0', 'G'],
]

# UNIVERSE = [
#     ['S','0','0','0',],
#     ['W','W','0','W',],
#     ['0','0','0','W',],
#     ['0','W','0','G',],
# ]
UNIT = 60  # size of each cell
# actions (you can remove each one you want)
ACTION_SPACE = ['U', 'R', 'D', 'L', 'UR', 'UL', 'DR', 'DL', 'NO']
UNIVERSE_SLEEP = 0   # game delay
REWARD_GOAL = 30   # reward of reaching the goal
REWARD_WALL = -20   # reward of falling into trap
REWARD_EMPTY = 0   # reward of going to empty cell


# UI class
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.n_action = len(ACTION_SPACE)  # Number of actions
        self.n_rows = len(UNIVERSE)  # Number of rows
        self.n_columns = len(UNIVERSE[0])   # Number of columns
        self.title('Maze')  # title of program
        self.geometry(f'{self.n_columns*UNIT}x{self.n_rows*UNIT}')
        self.origin = np.array([UNIT/2, UNIT/2])
        self._build_maze()

    # ----------- build UI from given universe ----------------

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=self.n_rows*UNIT,
                                width=self.n_columns*UNIT,
                                )

        # creating grids
        for i in range(0, self.n_rows*UNIT, UNIT):
            x0, y0, x1, y1 = 0, i, self.n_columns*UNIT, i
            self.canvas.create_line(x0, y0, x1, y1)
        for i in range(0, self.n_columns*UNIT, UNIT):
            x0, y0, x1, y1 = i, 0, i, self.n_rows*UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        self.walls = []
        padding = UNIT/2 - 5
        for i in range(self.n_rows):
            for j in range(self.n_columns):
                dst = np.array([j*UNIT, i*UNIT])
                if UNIVERSE[i][j] == 'W':
                    wall = self.canvas.create_rectangle(self.origin[0] + dst[0] - padding,
                                                        self.origin[1] +
                                                        dst[1] - padding,
                                                        self.origin[0] +
                                                        dst[0] + padding,
                                                        self.origin[1] +
                                                        dst[1] + padding,
                                                        fill='red',
                                                        )
                    self.walls.append(wall)
                if UNIVERSE[i][j] == 'G':
                    self.goal = self.canvas.create_oval(self.origin[0] + dst[0] - padding,
                                                        self.origin[1] +
                                                        dst[1] - padding,
                                                        self.origin[0] +
                                                        dst[0] + padding,
                                                        self.origin[1] +
                                                        dst[1] + padding,
                                                        fill='green',
                                                        )
                if UNIVERSE[i][j] == 'S':
                    self.start_point = dst

        self.marble = self.canvas.create_rectangle(self.origin[0] + self.start_point[0] - padding,
                                                   self.origin[1] +
                                                   self.start_point[1] -
                                                   padding,
                                                   self.origin[0] +
                                                   self.start_point[0] +
                                                   padding,
                                                   self.origin[1] +
                                                   self.start_point[1] +
                                                   padding,
                                                   fill='blue',
                                                   )

        self.canvas.pack()

    # ----------- reset universe --------------------------

    def reset(self):
        padding = UNIT/2 - 5
        self.update()
        time.sleep(UNIVERSE_SLEEP)
        self.canvas.delete(self.marble)
        self.marble = self.canvas.create_rectangle(self.origin[0] + self.start_point[0] - padding,
                                                   self.origin[1] +
                                                   self.start_point[1] -
                                                   padding,
                                                   self.origin[0] +
                                                   self.start_point[0] +
                                                   padding,
                                                   self.origin[1] +
                                                   self.start_point[1] +
                                                   padding,
                                                   fill='blue',
                                                   )

        return self.canvas.coords(self.marble)

    def step(self, action):
        s = self.canvas.coords(self.marble)
        movement = np.array([0, 0])

        if action == 'U':
            if s[1] > UNIT:
                movement[1] -= UNIT
        elif action == 'R':
            if s[0] < (self.n_columns - 1) * UNIT:
                movement[0] += UNIT
        elif action == 'D':
            if s[1] < (self.n_rows - 1) * UNIT:
                movement[1] += UNIT
        elif action == 'L':
            if s[0] > UNIT:
                movement[0] -= UNIT
        elif action == 'UR':
            if s[1] > UNIT:
                movement[1] -= UNIT
            if s[0] < (self.n_columns - 1) * UNIT:
                movement[0] += UNIT
        elif action == 'DR':
            if s[0] < (self.n_columns - 1) * UNIT:
                movement[0] += UNIT
            if s[1] < (self.n_rows - 1) * UNIT:
                movement[1] += UNIT
        elif action == 'DL':
            if s[1] < (self.n_rows - 1) * UNIT:
                movement[1] += UNIT
            if s[0] > UNIT:
                movement[0] -= UNIT
        elif action == 'UL':
            if s[0] > UNIT:
                movement[0] -= UNIT
            if s[1] > UNIT:
                movement[1] -= UNIT

        self.canvas.move(self.marble, movement[0], movement[1])
        s_next = self.canvas.coords(self.marble)

        # --------------- reward function -------------
        # reaching goal
        if s_next == self.canvas.coords(self.goal):
            reward = REWARD_GOAL
            done = True
        # falling into trap
        elif s_next in [self.canvas.coords(wall) for wall in self.walls]:
            reward = REWARD_WALL
            done = False
        else:
            reward = REWARD_EMPTY
            done = False

        return s_next, reward, done

    def render(self):
        self.update()
        time.sleep(UNIVERSE_SLEEP)

    def goal_coords(self):
        return self.canvas.coords(self.goal)

    def walls_coords(self):
        return [self.canvas.coords(wall) for wall in self.walls]

# %% [markdown]
# # 2.Reinforcement Learning

# %%
# RL constants:


EPSILON = 0.9
GAMMA = 0.8
ALPHA = 0.1
EPISODE = 1000


class RL:
    def __init__(self, goal, walls):
        self.q_table = pd.DataFrame(columns=ACTION_SPACE)
        self.goal = goal
        self.walls = walls

    def check_state_exist(self, state):
        if not state in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0.0]*len(ACTION_SPACE),
                    name=state,
                    index=self.q_table.columns
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.rand() < EPSILON:
            actions = self.q_table.loc[observation, :]
            actions = actions.reindex(np.random.permutation(actions.index))
            return actions.idxmax()
        return np.random.choice(ACTION_SPACE)

    def certain_action(self, observation):
        actions = self.q_table.loc[observation, :]
        actions = actions.reindex(np.random.permutation(actions.index))
        return actions.idxmax()

    def q_learn(self, state, action, reward, next_state):
        self.check_state_exist(next_state)

        if next_state == self.goal:
            self.q_table.loc[state, action] += ALPHA * \
                (reward - self.q_table.loc[state, action])
        else:
            self.q_table.loc[state, action] += ALPHA*(
                reward + GAMMA*self.q_table.loc[next_state, :].max() - self.q_table.loc[state, action])

    def sarsa_learn(self, state, action, reward, next_state, next_action):
        self.check_state_exist(next_state)

        if next_state == self.goal:
            self.q_table.loc[state, action] += ALPHA * \
                (reward - self.q_table.loc[state, action])
        else:
            self.q_table.loc[state, action] += ALPHA*(
                reward + GAMMA*self.q_table.loc[next_state, next_action] - self.q_table.loc[state, action])


# %% [markdown]
# # 3.run the program

# %%


def run_with_q_learing():
    global UNIVERSE_SLEEP
    m = Maze()
    agent = RL(m.goal_coords(), m.walls_coords())

    m.title('Maze - Training')

    for i in range(EPISODE):
        state = m.reset()
        state = str(state)
        step_counter = 0
        while True:
            action = agent.choose_action(state)
            clear_screen()
            print(f'action : {action}')
            print(agent.q_table)
            m.render()
            next_state, reward, done = m.step(action)
            next_state = str(next_state)
            agent.q_learn(state, action, reward, next_state)
            state = next_state
            step_counter += 1
            if done:
                break
        clear_screen()
        print(f'episode {i} have been done in {step_counter} steps')
        time.sleep(1)

    m.title('Maze - Solved')

    UNIVERSE_SLEEP = 1
    while True:
        state = m.reset()
        state = str(state)
        step_counter = 0
        while True:
            action = agent.certain_action(state)
            clear_screen()
            print(f'action : {action}')
            print(agent.q_table)
            m.render()
            next_state, reward, done = m.step(action)
            next_state = str(next_state)
            state = next_state
            step_counter += 1
            if done:
                break
        clear_screen()
        print(f'reach the goal in {step_counter} steps')
        time.sleep(1)


def run_with_sarsa():
    global UNIVERSE_SLEEP
    m = Maze()
    agent = RL(m.goal_coords(), m.walls_coords())

    m.title('Maze - Training')

    for i in range(EPISODE):
        state = m.reset()
        state = str(state)
        action = agent.choose_action(state)
        step_counter = 0
        while True:
            clear_screen()
            print(agent.q_table)
            m.render()
            next_state, reward, done = m.step(action)
            next_state = str(next_state)
            next_action = agent.choose_action(next_state)
            agent.sarsa_learn(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            step_counter += 1
            if done:
                break
        clear_screen()
        print(f'episode {i} have been done in {step_counter} steps')
        # time.sleep(1)

    m.title('Maze - Solved')

    UNIVERSE_SLEEP = 1
    while True:
        state = m.reset()
        state = str(state)
        action = agent.certain_action(state)
        step_counter = 0
        while True:
            clear_screen()
            print(f'action : {action}')
            print(agent.q_table)
            m.render()
            next_state, reward, done = m.step(action)
            next_state = str(next_state)
            next_action = agent.certain_action(next_state)
            # agent.sarsa_learn(state,action,reward,next_state,next_action)
            state = next_state
            action = next_action
            step_counter += 1
            if done:
                break
        clear_screen()
        print(f'reach the goal in {step_counter} steps')
        time.sleep(1)


# %%
run_with_sarsa()
