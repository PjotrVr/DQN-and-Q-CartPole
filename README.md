# About

This project is about showing how much of a difference experience replay memory and additional target network make when learning
even the simplest of the environments.

# Requirements

You'll only need: torch, numpy, matplotlib and gym.

Exact versions are available in `requirements.txt` file.

You can install those modules in command line/terminal using command: `pip install -r requirements.txt`

# Usage

Every main file can be run by writing in command line/terminal: `python3 main.py`

In case you don't want to use default parameters, you can switch them up, eg. `python3 main.py --lr 0.01`

List of parameters that you can change: <br>
                                        - `--lr` - learning rate <br>
                                        - `--discount` - discount factor for future rewards <br>
                                        - `--env` - name of the environment, currently only supports Gym environments <br>
                                        - `--capacity` - capacity for replay memory <br>
                                        - `--min_capacity` - amount of capacity that will be filled before training <br>
                                        - `--iterations` - number of iterations/episodes <br>
                                        - `--batch_size` - batch size used to sample replay memory <br>
                                        - `--target_freq_update` - frequency how often will target network take online network's parameters <br>
                                        - `--show_freq` - frequency how often will results be shown while training <br>
                                        - `--eps_start` - starting epsilon for epsilon-greedy exploration-exploitation policy <br>
                                        - `--eps_end` - ending epsilon for epsilon-greedy exploration-exploitation policy <br>
                                        - `--eps_decay` - amount how much will epsilon decrease while training <br>
                                        - `--save_model` - save model after training <br>
                                        - `--show_graphs` - show training graphs <br>

# Before training

Before training we can only solve this environment by using random action evey single time.

Obviously that won't work as we can see below.

<img src="CartPole Before Training Random Movement.gif"> </img>

# After training - Q learning

CartPole has 4 parameters in its observation space and it's also continuous. 

Since Q learning only works on discrete observation space we have to transform it by using bucket system with 21 buckets.

This means that we have too many states to learn and 15000 iterations aren't enough to solve this environment as we can see below.

<img src="CartPole Q Learning After Training.gif"> </img>

# After training - DQN learning

In DQN learning we don't have to have discrete observation space, we can approximate every single state from previous encounters.

Still that is the only upside of this method, it is also a lot slower than normal Q learning, but results aren't any better because it forgets everything after few iterations.

<img src="CartPole DQN Learning After Training.gif"> </img>

# After training - DQN with replay memory

With inclusion of experience replay memory we can start to see actual progress.

Biggest problem with this method is that we use same network for evaluating and predicting Q values.

<img src="CartPole DQN RM Learning After Training.gif"> </img>

# After training - DQN with target network and replay memory

This method with inclusion of target network really works!

Only downside is that it's slow, but considering that it actually only needed ~4000 iterations instead of 15000 we can say that it solved the environment.

<img src="CartPole DQN TN RM Learning After Training.gif"> </img>

# Results

This graph shows results of training.

As expected DQN with target network and replay memory performed the best and actually solved the environment (as shown in DQN learning gif).

<img src="Comparing Algorithms on CartPole.png">
