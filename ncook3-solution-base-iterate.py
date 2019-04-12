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

# This is a version of the `solution-base` notebook that allows for iterating automatically through multiple settings based on the values in an Excel spreadsheet.
#
# It is recommended that you fully understand `solution-base` before implementing this automated version.

# # Imports

# +
import cs7641assn4 as a4
import numpy as np
import pandas as pd
import warnings
import xlrd

pd.set_option('display.max_columns', 35)
# pd.reset_option("display.max_columns")

# +
# Default settings, uncomment to just use these
# settings = {'rH': [-1],
#  'rG': [1],
#  'rF': [-0.2],
#  'size': [4],
#  'p': [0.8],
#  'is_slippery': [False],
#  'render_initial': [True],
#  'epsilon': [1e-08],
#  'gamma': [0.8],
#  'max_iter': [10000],
#  'qepsilon': [0.1],
#  'lr': [0.8],
#  'qgamma': [0.95],
#  'episodes': [10000],
#  'initial': [1],
#  'decay': [True],
#  'report': [True],
#  'display_print': [True]}
# Import settings, uncomment to read settings from an excel spreadsheet
settings = pd.read_excel('settings_2019-04-12T1043.xlsx').to_dict()

# Determine the number of runs
n_settings = len(settings['rH'])


# +
for n in range(n_settings):
    # Load settings
    rH = settings['rH'][n] # -1 #-5 # reward for H(ole)
    rG = settings['rG'][n] #1 # 10 # reward for G(oal)
    rF = settings['rF'][n] #-0.2# reward includes S(tart) and F(rozen)
    size = settings['size'][n] #4 # height and width of square gridworld, [4, 8, 16] are included in cs7641assn4.py 
    p = settings['p'][n] #0.8 # if generating a random map probability that a grid will be F(rozen)
    map_name = 'x'.join([str(size)]*2) # None, if you want a random map
    desc = a4.MAPS[map_name] # None, if you want a random map
    is_slippery = settings['is_slippery'][n] #False
    render_initial = settings['render_initial'][n] # True

    epsilon = settings['epsilon'][n] #1e-8 # convergence threshold for policy/value iteration
    gamma = settings['gamma'][n] #0.8 # discount parameter for past policy/value iterations
    max_iter = settings['max_iter'][n] #10000 # maximum iterations for slowly converging policy/value iteration 

    # Qlearning(env, rH=0, rG=1, rF=0, qepsilon=0.1, lr=0.8, gamma=0.95, episodes=10000)
    qepsilon = settings['qepsilon'][n] #0.1 # epsilon value for the Q-learning epsilon greedy strategy
    lr = settings['lr'][n] #0.8 # Q-learning rate
    qgamma = settings['qgamma'][n] #0.95 # Q-Learning discount factor
    episodes = settings['episodes'][n] #10000 # number of Q-learning episodes
    initial = settings['initial'][n] #0 # value to initialize the Q grid
    decay = settings['decay'][n] #True

    # Printing options
    report = settings['report'][n] #True # For cs7641assn4.py policy and value iteration functions
    display_print = settings['display_print'][n] #True # For this script

    # Create Environment
    env = a4.getEnv(env_id='hw4-FrozenLake-v0', rH=rH, rG=rG, rF=rF, 
                    desc=desc,  
                    is_slippery=is_slippery, render_initial=True)

    # Store a representation of the map
    env_desc = env.desc.astype('<U8')

    # Store a representation of the state rewards
    env_rs = a4.getStateReward(env)

    if display_print:
        # Display reward at each state
        print('\n--Reward Values at Each State--')
        a4.matprint(a4.print_value(env_rs, width=size, height=size))
        
    ## Policy Iteration
    print('\n--Policy Iteration TimeIt--')
    pi_time = %timeit -o a4.policy_iteration(env, epsilon, gamma, max_iter, report=False)
    
    pi_V, pi_policy, pi_epochs = a4.policy_iteration(env, epsilon, gamma, max_iter, report=report)

    pi_policy_arrows = a4.print_policy(pi_policy, width=size, height=size)

    if display_print:
        # Display values
        print('\n--Policy Iteration Values in grid order--')
        a4.matprint(a4.print_value(pi_V, width=size, height=size))

        # Display policy
        print('\n--Policy Iteration Policy Matrix--')
        a4.matprint(pi_policy_arrows)
        
    ## Value Iteration
    print('\n--Value Iteration TimeIt--')
    vi_time = %timeit -o a4.valueIteration(env, epsilon, gamma, max_iter, report=False)
    
    vi_V, vi_epochs = a4.valueIteration(env, epsilon, gamma, max_iter, report=report)

    vi_policy = a4.value_to_policy(env, V=vi_V, gamma=gamma)

    vi_policy_arrows = a4.print_policy(vi_policy, width=size, height=size)

    if display_print:
        # display value function:
        print('\n--Value Iteration Values in grid order--')
        a4.matprint(a4.print_value(vi_V, width=size, height=size))
        
        # display policy
        print('\n--Value Iteration Policy Matrix--')
        a4.matprint(vi_policy_arrows)
        
    ## Q-Learning
    print('\n--Q-Learning TimeIt--')
    Q_time = %timeit -o a4.Qlearning(env, qepsilon, lr, qgamma, episodes, initial, decay, report=False)
        
    Q, Q_epochs = a4.Qlearning(env, qepsilon, lr, qgamma, episodes, initial, decay, report)

    maxQ = np.max(Q,axis=1)

    Q_policy = a4.Q_to_policy(Q)

    Q_policy_arrows = a4.print_policy(Q_policy, width=size, height=size)

    if display_print: 
        print('--Q with all options--')
        a4.matprint(Q)
        print('\n--argmax(Q) in grid order--')
        a4.matprint(a4.print_value(maxQ, width=size, height=size))
        print('\n--Q-Learning Policy Matrix--')
        a4.matprint(Q_policy_arrows)
        
    ## Save results to DataFrame
    results = pd.DataFrame({'rH': [rH], 
                        'rG': [rG], 
                        'rF': [rF], 
                        'size': [size], 
                        'p': [p], 
                        'desc': [desc], 
                        'map_name': [map_name],                        
                        'is_slippery': [is_slippery],
                        'epsilon': [epsilon],
                        'gamma': [gamma], 
                        'max_iter': [max_iter], 
                        'qepsilon': [qepsilon], 
                        'lr': [lr], 
                        'qgamma': [qgamma], 
                        'episodes': [episodes], 
                        'initial': [initial],
                        'env_desc': [env_desc],
                        'env_rs': [env_rs],
                        'pi_time': [pi_time.average],
                        'pi_V': [pi_V],
                        'pi_epochs': [pi_epochs],
                        'pi_policy': [pi_policy],
                        'pi_policy_arrows': [pi_policy_arrows],
                        'vi_time': [vi_time.average],
                        'vi_V': [vi_V],
                        'vi_epochs': [vi_epochs],
                        'vi_policy': [vi_policy],
                        'vi_policy_arrows': [vi_policy_arrows],
                        'Q_time': [Q_time.average],
                        'Q': [Q],
                        'Q_epochs': [Q_epochs],
                        'Q_V': [maxQ],
                        'Q_policy': [Q_policy],
                        'Q_policy_arrows': [Q_policy_arrows]})
    
    if display_print: 
        display(results)
        
    ## Save results to disk
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    try:
        dataset = pd.read_hdf('data.h5', key='dataset', mode='a')
    except FileNotFoundError:
        results.to_hdf('data.h5', key='dataset', mode='a')
    else:
        dataset.append(
            other=results, 
            ignore_index=True,
            sort=False
            ).to_hdf(
            path_or_buf='data.h5', 
            key='dataset', 
            mode='a')
        
    if display_print:
        pd.read_hdf('data.h5', key='dataset', mode='a')
    
    
print('Complete!')   
# -

# # Notes

# Default rewards in OpenAI gym Frozen-Lake-v0 are 1 for the G(oal) and 0 for everything else.
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

# - Environment: <https://gym.openai.com/envs/FrozenLake-v0/>
# - Code: <https://github.com/Twice22/HandsOnRL>
# - Tutorial: <https://twice22.github.io/>
