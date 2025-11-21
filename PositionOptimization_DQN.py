import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import collections
import random

# ------------------------------------- #
# 经验回放池
# ------------------------------------- #
class DQN_ReplayBuffer:
    def __init__(self, capacity): # 经验池的最大容量
        # 创建一个队列，先进先出
        self.buffer = collections.deque(maxlen = capacity)

    # 在队列中添加数据
    def add(self, state, action, reward, next_state, done):
        # 以list类型保存
        self.buffer.append((state, action, reward, next_state, done))

    # 在队列中随机取样batch_size组数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将数据集拆分开来
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 测量当前时刻的队列长度
    def size(self):
        return len(self.buffer)

# 深度网络
class Net(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(Net, self).__init__()
        # 只包含一个隐藏层
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, n_actions)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class DQN:
    def __init__(self, n_states, n_hiddens, n_actions, lr, tau, gamma, device):
        # 属性分配
        self.n_states = n_states
        self.n_actions = n_actions
        self.tau = tau
        self.gamma = gamma
        self.device = device

        # 评估网络
        self.eval_net = Net(n_states, n_hiddens, n_actions).to(device)
        # 现实网络
        self.target_net = Net(n_states, n_hiddens, n_actions).to(device)
        # 同步参数
        self.target_net.load_state_dict(self.eval_net.state_dict())

        # 损失函数
        self.loss = nn.MSELoss()
        # 优化器，优化评估神经网络
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = lr)

        self.cost = []  # 记录损失值

    # 动作选择
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)
        # 策略网络计算出当前状态下的动作价值
        action_value = self.eval_net.forward(x).squeeze().detach().cpu().numpy()
        return action_value

    # 给动作加入噪声，增加探索空间
    def add_noise_to_weights(self, weights, sigma=0.1):
        noise = torch.normal(mean=0, std=sigma, size=weights.shape).squeeze().detach().cpu().numpy()
        return weights + noise

    # 软更新，意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    # 从存储学习数据
    def learn(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # q_eval的学习过程
        q_eval = self.eval_net(states)
        # 下一步状态的预测：
        q_next = self.target_net(next_states).detach().max(1, keepdim=True)[0]
        q_target = rewards + self.gamma * q_next * (1 - dones)
        # 通过预测值与真实值计算损失 q_eval预测值， q_target真实值
        loss = self.loss(q_eval, q_target.expand_as(q_eval))
        self.cost.append(loss.squeeze().detach().cpu().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新策略网络的参数
        self.soft_update(self.eval_net, self.target_net)