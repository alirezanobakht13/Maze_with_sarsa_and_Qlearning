# Maze_with_sarsa_and_Qlearning
Solving maze using SARSA and Q learning methods.

based on book:
> Reinforcement Learning  With Open AI, TensorFlow and Keras Using Python

written by Abhishek Nandy, Manisha Biswas

**Required packages**:
* numpy
* pandas
* tkinter

## there are two major parts:
1. **creating universe**
2. **Reinforcement Learning implementation** 

in each part there are constants that you can change them to see differences

## Creating Universe:
* **UNIVERSE:** two-d array that map of game is created based on. you can add or delete cells or change their value:(characters are upper case.)
    * **'S':** starting point
    * **'G':** destination
    * **'W':** wall
    * **'0':** empty cell(character is zero)
* **Rewards:**
    * **REWARD_GOAL**: Reward of reaching the goal
    * **REWARD_WALL**: Reward(punishment) falling into trap. (should be negetive value).
    * **REWARD_EMPTY**: Reward of going into empty cell. (could be negetive or zero)
* **ACTION_SPACE:** array of available actions(movement directions). you can delete some of them and see changes.
    * **'U'**: Up
    * **'R'**: Right
    * **'D'**: Down
    * **'L'**: Left
    * **'UR'**: Up Right
    * **'UL'**: UP Left
    * **'DR'**: Down Right
    * **'DL'**: Down Left
    * **'NO'**: No Action
* **UNIT**: width and height of each cell in UI
* **UNIVERSE_SLEEP**: delay between steps.

## Reinforcement Learning:
* **EPSILON**: ϵ parameter of ϵ-greedy algorithms. (probability of picking random action in each step).
* **GAMMA**: discount factor. (should be between 0 and 1, indicate how much we value feature rewards).
* **ALPHA**: Learning rate.
* **EPISODE**: number of time that agent plays game for learning.

## Run:
at the end of the code one add one of these two functions:
* `run_with_sarsa()`: agent will learn and play game using **SARSA** algorithm.
* `run_with_q_learing()`: agent will learn and play game using **Q Learning** algorithm.

and then run the code.


