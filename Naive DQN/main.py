import argparse
import os

import numpy as np
import gym

from agent import Agent


# Creating option for command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--env", help="Name of environment", default="CartPole-v0")
parser.add_argument("--iterations", help="Number of iterations/episodes", default=10000)
parser.add_argument("--lr", help="Learning rate", default=1e-3)
parser.add_argument("--discount", help="Discount factor", default=0.99)
parser.add_argument("--eps_start", help="Epsilon start", default=1)
parser.add_argument("--eps_end", help="Epsilon end", default=0.01)
parser.add_argument("--eps_decay", help="Epsilon decay", default=1e-5)
parser.add_argument("--show_freq", help="Frequency for showing results", default=100)
parser.add_argument("--show_graphs", help="Show graphs for loss, epsilon and reward", default=True) #TODO
parser.add_argument("--save_model", help="Save model", default=True) #TODO

args = parser.parse_args()

env = gym.make(args.env)

# Hyperparams
NUM_ITERATIONS = args.iterations
DISCOUNT = args.discount
LR = args.lr
EPSILON_START = args.eps_start
EPSILON_END = args.eps_end
EPSILON_DECAY = args.eps_decay
SHOW_FREQ = args.show_freq

# Creating an agent
agent = Agent(env, lr=LR, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY, discount=DISCOUNT)

epsilon_history, score_history = [], []
for iteration in range(NUM_ITERATIONS):
    state = agent.env.reset()
    done = False
    score = 0
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = agent.env.step(action)
        
        # Iteration through DQN
        agent.learn(state, action, reward, next_state)
        
        state = next_state
        score += reward
    
    if iteration % SHOW_FREQ == 0 and iteration != 0:
        average_score = np.mean(score_history[-SHOW_FREQ:])
        print(f"Episode: {iteration}, Score: {score}, Average Score: {average_score}, Epsilon: {agent.epsilon}")
        
    # Add results to history
    score_history.append(score)
    epsilon_history.append(agent.epsilon)