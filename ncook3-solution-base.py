# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Imports

import cs7641assn4 as a4
from numpy import argmax

# # Establish Environment

# +
id = 'Deterministic-4x4-FrozenLake-v0' # string identifier for environment, arbitrary label
rH = -1 #-5 # reward for H(ole)
rG = 1 # 10 # reward for G(oal)
rF = -0.2# reward includes S(tart) and F(rozen)
size = 4 # height and width of square gridworld
p = 0.8 # if generating a random map probability that a grid will be F(rozen)
desc = None # frozen_lake.generate_random_map(size=size, p=p)
map_name = 'x'.join([str(size)]*2) # None
is_slippery = False


epsilon = 1e-8 # convergence threshold for policy/value iteration
gamma = 0.8 # discount parameter for past policy/value iterations
max_iter = 10000 # maximum iterations for slowly converging policy/value iteration 

# Qlearning(env, rH=0, rG=1, rF=0, qepsilon=0.1, lr=0.8, gamma=0.95, episodes=10000)
qepsilon = 0.1 # epsilon value for the Q-learning epsilon greedy strategy
lr = 0.8 # Q-learning rate
qgamma = 0.95 # Q-Learning discount factor
episodes = 10000 # number of Q-learning episodes

# Create Environment
env = a4.getEnv(id=id, rH=rH, rG=rG, rF=rF, desc=desc, map_name=map_name, is_slippery=is_slippery,render_initial=True)

# Display reward at each state
print('\n--Reward Values at Each State--')
a4.matprint(a4.print_value(a4.getStateReward(env),width=size,height=size))
# -

# # Policy Iteration

# ```
# # %%timeit
# V, pi, epochs = a4.policy_iteration(env, epsilon=epsilon, gamma=gamma, max_iter=max_iter, report=False)
# ```
#
# 670 µs ± 55.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

# +
V, pi, epochs = a4.policy_iteration(env, epsilon=epsilon, gamma=gamma, max_iter=max_iter, report=True)

# Display values
a4.matprint(a4.print_value(V))

# Display policy
a4.matprint(a4.print_policy(pi, width=size, height=size))
# -

# # Value Iteration

# ```
# # %%timeit
# V, epochs = a4.valueIteration(env, epsilon=epsilon, gamma=gamma, max_iter=max_iter, report=False)
# ```
#
# 6.38 ms ± 658 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# +
V, epochs = a4.valueIteration(env, epsilon=epsilon, gamma=gamma, max_iter=max_iter, report=True)

# display value function:
a4.matprint(a4.print_value(V))

pol = a4.value_to_policy(env, V=V, gamma=gamma)

# display policy
a4.matprint(a4.print_policy(pol, width=size, height=size))
# -

# # Q-Learning

# +
Q = a4.Qlearning(env, qepsilon, lr, qgamma, episodes)
print('--Q with all options--')
a4.matprint(Q)

maxQ = Q[range(len(Q)),argmax(Q,axis=1)]
print('\n--argmax(Q) in grid order--')
a4.matprint(a4.print_value(maxQ))

s, steps = a4.Qlearning_trajectory(env, Q, render=False)
print('\n--Policy Matrix--')
a4.matprint(a4.policy_matrix(Q))
# -

# # Notes

# Default rewards are 1 for the G(oal) and 0 for everything else.
#
# Maps are drawn according to the following logic
#
# ```
# if desc and map_name are None, 
#    then a default random map is drawn with 8
#         using frozen_lake.generate_random_map(size=8, p=0.8)
# elif desc is None and a map_name is given
#    then a map_name is either '4x4' or '8x8'
#         and is drawn from the dict MAPS in frozen_lake.py
# elif desc is given
#    then it must be in the form of a list with 
# ```
#
# Default action probabilities are 1/3 chosen action, 1/3 each for right angles to chosen action, and 0 for reverse of chosen action. This is set with `is_slippery=True`. If `is_slippery=False`, then P=1 for chosen action and 0 for all other actions.
#
# |ACTION|Value|Symbol|
# |------|-----|------|
# |LEFT  | 0   | ←    |
# |DOWN  | 1   | ↓    |
# |RIGHT | 2   | →    |
# |UP    | 3   | ↑    |

# # Sources

# - Code: <https://github.com/Twice22/HandsOnRL>
# - Tutorial: <https://twice22.github.io/>
