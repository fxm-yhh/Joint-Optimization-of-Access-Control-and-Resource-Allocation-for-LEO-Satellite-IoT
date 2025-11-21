import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random
import matplotlib.pyplot as plt


# ------------------------------------- #
# 经验回放池
# ------------------------------------- #
class DDPG_ReplayBuffer:
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

# ------------------------------------- #
# 策略网络
# ------------------------------------- #
class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        # 只包含一个隐含层
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

# ------------------------------------- #
# 价值网络
# ------------------------------------- #
class QValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(QValueNet, self).__init__()
        #
        self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)

    def forward(self, x, a):
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)
        x = self.fc1(cat)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

# ------------------------------------- #
# 算法主体
# ------------------------------------- #
class DDPG:
    def __init__(self, n_states, n_hiddens, n_actions, actor_lr, critic_lr, tau, gamma, device):
        # 策略网络--训练
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 价值网络--训练
        self.critic = QValueNet(n_states, n_hiddens, n_actions).to(device)
        # 策略网络--目标
        self.target_actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 价值网络--目标
        self.target_critic = QValueNet(n_states, n_hiddens, n_actions).to(device)

        # 初始化价值网络的参数，两个价值网络的参数相同
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化策略网络的参数，两个策略网络的参数相同
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 属性分配
        self.gamma = gamma # 折扣因子
        self.tau = tau  # 目标网络的软更新参数
        self.n_actions = n_actions
        self.device = device

        # 用于画图，观察性能的参数
        self.actor_cost = []

    # 动作选择
    def take_action_weights(self, state):
        state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)
        # 策略网络计算出当前状态下的动作价值
        action_weights = self.actor(state).squeeze().detach().cpu().numpy()
        return action_weights

    # 给动作加入噪声，增加探索空间
    def add_noise_to_weights(self, weights, sigma=0.1):
        # 在权重上添加高斯噪声
        noise = torch.normal(mean=0, std=sigma, size=weights.shape).squeeze().detach().cpu().numpy()
        return weights + noise

    # 软更新，意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data*(1-self.tau) + param.data*self.tau)

    # 训练
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # ------------------------------------- #
        # Critic网络更新
        # ------------------------------------- #
        # 价值目标网络获取下一时刻的动作
        next_q_values = self.target_actor(next_states)
        # 策略目标网络获取下一时刻状态选出的动作价值
        next_q_values = self.target_critic(next_states, next_q_values)
        # 当前时刻的动作价值的目标值
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # 当前时刻动作价值的预测值
        q_values = self.critic(states, actions)

        # 预测值和目标值之间的均方差损失
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # 价值网络梯度
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------------- #
        # Actor网络更新
        # ------------------------------------- #
        # 当前状态的每个动作的价值
        actor_q_values = self.actor(states)
        # 当前状态选出的动作价值
        score = self.critic(states, actor_q_values)
        # 计算损失
        actor_loss = -torch.mean(score)
        self.actor_cost.append(actor_loss.detach().cpu().numpy())
        # 策略网络梯度
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------- #
        # target_actor和target_critic网络更新
        # ------------------------------------- #
        # 软更新策略网络的参数
        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络的参数
        self.soft_update(self.critic, self.target_critic)

    # 绘制性能图
    def plot_actor_cost(self):
        plt.plot(np.arange(len(self.actor_cost)), self.actor_cost, linewidth=2.0, label='DDPG Network')
        # 保存actor_cost
        df = pd.DataFrame(self.actor_cost)
        df.to_csv(r'E:\PYTHON\RandomAccess_DeepLearn\Data_toorigin\actor_cost.csv', index=False)
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.xlabel("Iteration")
        plt.ylabel("Loss function")
        plt.legend()
        plt.savefig(r'E:\PYTHON\RandomAccess_DeepLearn\Data_toorigin\figure_cost.pdf', format='pdf')
        plt.show()
