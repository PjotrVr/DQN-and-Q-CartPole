import torch
import numpy as np

from dqn import DQN


class Agent:
    def __init__(self, env, lr, epsilon_start, epsilon_end, epsilon_decay, discount):
        self.env = env
        self.state_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.action_space = [action for action in range(self.num_actions)]
        
        self.lr = lr
        self.discount = discount
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_online = DQN(self.state_shape, self.num_actions, self.lr)
        
    def choose_action(self, observation):
        # We use epsilon-greedy policy
        if np.random.random() >= self.epsilon:
            # In case that device is CUDA we have to sent it to GPU instead of CPU, also we have to transform it to torch tensor
            state = torch.tensor([observation], dtype=torch.float).to(self.q_online.device)
            with torch.no_grad():
                actions = self.q_online(state)
            # Take action with the highest value
            action = torch.argmax(actions).item()
        
        else:
            # Take random action in action space, eg. action space is [0, 1, 2] it takes 2nd action
            action = np.random.choice(self.action_space)
            
        return action
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
      
    def learn(self, state, action, reward, next_state):
        # Zero the gradients
        self.q_online.optimizer.zero_grad()
        
        # Turning numpy arrays into torch tensors
        state_t = torch.tensor(state, dtype=torch.float).to(self.q_online.device)
        action_t = torch.tensor(action, dtype=torch.int).to(self.q_online.device)
        reward_t = torch.tensor(reward, dtype=torch.float).to(self.q_online.device)
        next_state_t = torch.tensor(next_state, dtype=torch.float).to(self.q_online.device)
        
        # Estimating Q value
        q_pred = self.q_online(state_t)[action_t]
        q_next = self.q_online(next_state_t).max()
        q_target = reward + self.discount * q_next
        
        # Calculating loss
        loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
        
        # Backprop
        loss.backward()
        self.q_online.optimizer.step()
        
        # Decrease epsilon
        self.update_epsilon()