import numpy as np


class ReplayMemory:
    def __init__(self, capacity, input_shape):
        self.capacity = capacity
        self.memory_counter = 0

        self.state_memory = np.zeros((self.capacity, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.capacity, dtype=np.int64)
        self.reward_memory = np.zeros(self.capacity, dtype=np.float32)
        self.next_state_memory = np.zeros((self.capacity, *input_shape), dtype=np.float32)
        self.terminal_memory = np.zeros(self.capacity, dtype=np.uint8)

    def store(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.capacity
        
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def sample(self, batch_size):
        max_memory = min(self.memory_counter, self.capacity)
        batch = np.random.choice(max_memory, batch_size, replace=False)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return min(self.memory_counter, self.capacity)