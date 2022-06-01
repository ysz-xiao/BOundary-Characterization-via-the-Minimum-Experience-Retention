import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

device = "cpu"

# hyper-parameters
# 参数设置：https://www.saashanair.com/dqn-hyperparameter-experiments/
BATCH_SIZE = 128
LR = 0.001
GAMMA = 0.99
EPISILO_MIN = 0.01
EPISILO_MAX = 0.99
EPISILO_DECAY = 0.995
MEMORY_CAPACITY = 10000
Q_NETWORK_ITERATION = 1000


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self, NUM_STATES, NUM_ACTIONS, ENV_A_SHAPE):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, NUM_ACTIONS)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN():
    """docstring for DQN"""

    def __init__(self, NUM_STATES, NUM_ACTIONS, ENV_A_SHAPE):
        super(DQN, self).__init__()
        self.NUM_STATES = NUM_STATES
        self.NUM_ACTIONS = NUM_ACTIONS
        self.ENV_A_SHAPE = ENV_A_SHAPE
        self.eval_net = Net(self.NUM_STATES, self.NUM_ACTIONS, self.ENV_A_SHAPE).to(device)
        self.target_net = Net(self.NUM_STATES, self.NUM_ACTIONS, self.ENV_A_SHAPE).to(device)
        self.episilo = EPISILO_MAX
        self.episilo_decay = EPISILO_DECAY
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter = 0
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def save_to_desk(self, path, file_name):
        """存储模型"""
        torch.save(self, path + file_name + ".model")

    def load_from_desk(self, path, file_name):
        """加载模型"""
        torch.load(path + file_name + ".model")

    def forward(self, state, train=True):
        state = torch.unsqueeze(torch.FloatTensor(state).to(device), 0)  # get a 1D array
        EPISILO = self.episilo if self.episilo > EPISILO_MIN else EPISILO_MIN
        if (train == True) and (np.random.rand() <= EPISILO):  # random policy
            action = np.random.randint(0, self.NUM_ACTIONS)
            action = action if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
        else:  # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
        return action

    def store_transition(self, trans):
        self.memory.append(trans)

    def learn(self):
        if len(self.memory) < BATCH_SIZE * 2:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*(zip(*batch)))
        batch_state = torch.FloatTensor(batch.state).to(device)
        batch_action = torch.LongTensor(batch.action).to(device).unsqueeze(1)
        batch_reward = torch.FloatTensor(batch.reward).to(device)
        batch_next_state = torch.FloatTensor(batch.next_state).to(device)
        batch_done = torch.FloatTensor(batch.done).to(device)

        # q_eval
        # torch.detach()返回一个新的Variable，从当前计算图中分离下来，但是不需要计算其梯度，requires_grad为true
        q_eval = self.eval_net(batch_state).gather(1, batch_action).squeeze()
        q_next = self.target_net(batch_next_state).max(1)[0]
        q_target = batch_reward + GAMMA * (1 - batch_done) * q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
