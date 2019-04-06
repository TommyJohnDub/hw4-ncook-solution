import gym.envs.toy_text.frozen_lake as frozen_lake # I guess `toy_text` is a separate package?
from gym import make
from gym.envs.registration import register, registry
import numpy as np
import math
import my_env

###
# Utility Functions
###

def getStateReward(env):
    n_states = env.observation_space.n
    Rs = np.empty(n_states)
    Rs.fill(np.nan)
    p = env.P
    for state in p:
        for action_commanded in p[state]:
            for action_possible in p[state][action_commanded]:
                Rs[action_possible[1]] = action_possible[2]
    
    return Rs    

def getReward(env):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    
    R = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a, moves in env.P[s].items():
            for possible_move in moves:
                prob, _, r, _ = possible_move
                R[s, a] += r * prob
    
    return R

def getProb(env):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    
    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            for moves in env.P[s][a]:
                prob, next_s, _, _ = moves
                P[s, a, next_s] += prob
    
    return P

def print_value(V, width=4, height=4):
    return np.around(np.resize(V, (width, height)), 4)

# let's plot the policy matrix (as in Part 1). according to
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
# LEFT = 0   DOWN = 1   RIGHT = 2  UP = 3
def print_policy(V, width=4, height=4):
    table = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    policy = np.resize(V, (width, height))
    
    # transform using the dictionary
    return np.vectorize(table.get)(policy)

# https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
# matprint.py Pretty print a matrix in Python 3 with numpy
def matprint(mat, fmt="g"):
    if mat[0][0] in ["←", "↓", "→", "↑"]:
        fmt = 's'# fmt='s' for arrows
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
        
        
###
# Policy Iteration Functions
###
        
# to evaluate the policy, as there is no max in the equation we can just solve
# the linear system
def policy_evaluation(pi, P, R, gamma, n_states):
    p = np.zeros((n_states, n_states))
    r = np.zeros((n_states, 1))
    
    for s in range(n_states):
        r[s] = R[s, pi[s]]
        p[s, :] = P[s, pi[s], :]
    
    # we take [:, 0] to return a vector because otherwise we have
    # a matrix of size (# states, 1)
    return np.linalg.inv((np.eye(n_states) - gamma * p)).dot(r)[:, 0]

def policy_iteration(env, epsilon=1e-8, gamma=0.8, max_iter=10000, report=False):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    
    # initialize arbitrary value function
    V = np.zeros(n_states)
    
    # initialize arbitrary policy
    pi = np.ones(n_states, dtype=int)
    
    R = getReward(env)
    P = getProb(env)
    
    i = 0
    
    while True and i < max_iter:
        V_prev = V.copy()
        
        # evaluate the policy
        V = policy_evaluation(pi, P, R, gamma, n_states)
        
        # policy improvement
        for s in range(n_states):
            pi[s] = np.argmax(R[s,:] + gamma * P[s, :, :].dot(V)) 
        
        if np.linalg.norm(V_prev - V) < epsilon:
            if report:
                print("Policy iteration converged after ", i+1, "epochs")
            break
        
        i += 1
    
    return V, pi, i+1


###
# Value Iteration Functions
###
def valueIteration(env, epsilon, gamma, max_iter=10000, report=True):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    
    # initialize utilities to 0
    V = np.zeros(n_states)
    
    R = getReward(env)
    P = getProb(env)
    
    i = 0
    while True and i < max_iter:
        i += 1
        prev_V = V.copy()
        for s in range(n_states):
            V[s] = max(R[s,:] + gamma * P[s, :, :].dot(V))

        if np.linalg.norm(prev_V - V) <= epsilon:
            if report:
                print("Value iteration converged after ", i+1, "epochs")
            break
    
    return V, i+1

# transform value function into a policy
def value_to_policy(env, gamma, V):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    
    policy = np.zeros(n_states, dtype=int)
    for state in range(n_states):
        best_action = 0
        best_reward = -float("inf")
        for action in range(n_actions):
            moves = env.P[state][action] # [(prob, next_state, reward, terminate), ...]
            avg_reward = sum([prob * reward + gamma * V[next_state] for (prob, next_state, reward, _) in moves])
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = action
        
        policy[state] = best_action
    
    return policy


###
# Generate a customized frozen lake
###

def getEnv(id='default', rH=0, rG=1, rF=0, desc=None, map_name='4x4', is_slippery=True, render_initial=True):
    all_envs = registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if id not in env_ids:
        register(
            id=id, # name given to this new environment
            entry_point='my_env:CustomizedFrozenLake', # env entry point
            kwargs={'rH': rH, 'rG': rG, 'rF': rF, 
                    'desc': desc,
                    'map_name': map_name, 
                    'is_slippery': is_slippery} # argument passed to the env
        )

    this_env = make(id)

    if render_initial:
        this_env.render()
        display(this_env.P[this_env.nS - 1][0])
    
    return this_env

# Random nugget
# env.unwrapped.spec.id # https://stackoverflow.com/questions/52774793/get-name-id-of-a-openai-gym-environment