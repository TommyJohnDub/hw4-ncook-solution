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

# # Establish Environment

# +
id = 'Deterministic-4x4-FrozenLake-v0' # string identifier for environment, arbitrary label
rH = 0 #-5 # reward for H(ole)
rG = 1 # 10 # reward for G(oal)
rF = 0 # reward includes S(tart) and F(rozen)
size = 4 # height and width of square gridworld
p = 0.8 # if generating a random map probability that a grid will be F(rozen)
desc = None # frozen_lake.generate_random_map(size=size, p=p)
map_name = 'x'.join([str(size)]*2) # None
is_slippery = False

epsilon = 1e-8 # convergence threshold for policy/value iteration
gamma = 0.8 # discount parameter for past policy/value iterations
max_iter = 10000 # maximum iterations for slowly converging policy/value iteration 

# Create Environment
env = a4.getEnv(id=id,render_initial=True)

# Display reward at each state
a4.matprint(a4.print_value(a4.getStateReward(env)))
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

# %%timeit
V = a4.valueIteration(env, epsilon=epsilon, gamma=gamma, max_iter=max_iter, report=False)

# +
V, epochs = a4.valueIteration(env, epsilon=epsilon, gamma=gamma, max_iter=max_iter, report=True)

# display value function:
a4.matprint(a4.print_value(V))

pol = a4.value_to_policy(env, gamma=gamma, V=V)

# display policy
a4.matprint(a4.print_policy(pol, width=size, height=size))
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
# |LEFT|0|←|
# |DOWN | 1|↓|
# |RIGHT | 2|→|
# |UP | 3| ↑|

# # Sources

# - Code: <https://github.com/Twice22/HandsOnRL>
# - Tutorial: <https://twice22.github.io/>
