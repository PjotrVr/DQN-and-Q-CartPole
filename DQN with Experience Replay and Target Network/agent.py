import torch
import numpy as np
from dqn import DQN
from experience_replay import ReplayMemory


class Agent:
    def __init__(self, env, lr, batch_size, epsilon_start, epsilon_end, epsilon_decay, discount, memory_size, min_memory_size, target_freq_update):
        self.env = env
        self.state_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.action_space = [action for action in range(self.num_actions)]
        
        self.lr = lr
        self.batch_size = batch_size
        self.discount = discount
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_freq_update = target_freq_update
        
        # Create experience replay buffer
        self.memory = ReplayMemory(memory_size, self.state_shape)
        # Fill minimal amount of experience needed before the training
        self.fill_min_memory(min_memory_size)
        
        self.q_online = DQN(self.state_shape, self.num_actions, self.lr)
        self.q_target = DQN(self.state_shape, self.num_actions, self.lr)
        
        # Target and online network need to have same weights
        self.update_target_network()
        
    def fill_min_memory(self, min_memory_capacity):
        state = self.env.reset()
        done = False
        for _ in range(min_memory_capacity):
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.store(state, action, reward, next_state, done)
            state = next_state

            if done:
                state = self.env.reset()
                done = False

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
    
    def update_target_network(self):
        self.q_target.load_state_dict(self.q_online.state_dict())
    
    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
        
    def sample(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Transform all numpy arrays into torch tensors
        states_t = torch.tensor(states).to(self.q_online.device)
        actions_t = torch.tensor(actions).to(self.q_online.device)
        rewards_t = torch.tensor(rewards).to(self.q_online.device)
        next_states_t = torch.tensor(next_states).to(self.q_online.device)
        dones_t = torch.tensor(dones).to(self.q_online.device)
        
        return states_t, actions_t, rewards_t, next_states_t, dones_t
    
    def learn(self):
        # In case that we don't have enough memory than don't learn
        if self.memory.memory_counter < self.batch_size:
            return
        # Zero the gradients
        self.q_online.optimizer.zero_grad()
        
        # Take sample of replay memory
        states, actions, rewards, next_states, dones = self.sample()
        
        indicies = np.arange(self.batch_size)
        q_pred = self.q_online(states)[indicies, actions]
        q_next = self.q_target(next_states).max(dim=1)[0]
        q_next[dones] = 0.0
        
        # Recalculating Q value
        q_target = rewards * self.discount * q_next
        
        # Calculating loss
        loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
        
        # Backprop
        loss.backward()
        self.q_online.optimizer.step()
        
        # Decrease epsilon
        self.update_epsilon()