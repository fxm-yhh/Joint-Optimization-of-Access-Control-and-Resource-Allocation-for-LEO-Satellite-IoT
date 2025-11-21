import numpy as np
import pandas as pd
import torch
import math
from SatelliteIoT_env import SatelliteIoT, SharedData
from PositionOptimization_DDPG import DDPG_ReplayBuffer, DDPG
from PositionOptimization_DQN import DQN_ReplayBuffer, DQN

## 场景说明
# 这是对DDPG网络进行预训练的模块，以便于在"ControlCentre.py"中可以直接使用DDPG
## END
## 程序说明
#
## END

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 超参数
IS_PRE_TRAINING = True # 指示是否是预训练
EPISODE_LOOP = 20 # 训练周期个数
SATELLITE_TIME = 200 # 每个卫星覆盖的时间
ACTIONS_SPACE = 10 # 卫星可以分配的不同频点子信道数
TIME_SLOT = 10 # 每个信道的时隙块数
FRAME_LENGTH = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # 地面设备可选的总帧长
EQUIPMENT_ALPHA = 100 # 地面设备数量折扣因子
BUFFER_SIZE = 5000 # DQN和DDPG记忆库的长度
BATCH_SIZE = 64 # DQN和DDPG记忆库采样个数
IS_DDPG = True # 选择使用DDPG还是DQN
TO_SAVE_MODEL = False # 在此次训练中是否保存训练好的模型

# 用DQN网络决策的
def satellite_update_dqn():
    # time of communication between satellite and ground equipment
    for satellite_moment in range(SATELLITE_TIME):
        # 检查地面设备是否全部发送完数据
        print('**第' + str(episode) + '个周期', '，第' + str(satellite_moment) + '个卫星时刻')
        if env.check_ending(satellite_moment):
            break

        # 地面设备发送信息帧至卫星
        tf_maps, receive_information = env.equipment_send_sonframe()

        # 卫星侧接收到信号之后对时频图进行分析，对信息帧位置进行优化，以达到最大的利用率
        # ------------------------------------- #
        # 深度强化学习DQN方法
        # ------------------------------------- #
        state_row1 = np.array([sum(row) for row in tf_maps])
        state_row2 = np.array(TIME_SLOT - state_row1)
        state = np.concatenate((state_row1, state_row2), 0)

        # 获取当前状态对应的动作
        action_weights = dqn_agent.choose_action(state)  # 动作权重
        sigma = max(0.01, 1.0 - episode / 2 * 0.01)  # 每个循环减小噪声强度
        noisy_weights = dqn_agent.add_noise_to_weights(action_weights, sigma=sigma)
        actions = env.transfer_action(noisy_weights)

        # 环境更新
        next_state, reward, done = env.update_step(state_row2, actions)

        # 更新经验回放池
        dqn_replay_buffer.add(state, action_weights, reward, next_state, done)

        # 如果经验池超过容量，开始训练
        if dqn_replay_buffer.size() > int(BUFFER_SIZE / 10):
            # 经验池随机采样batch_size组
            s, a, r, ns, d = dqn_replay_buffer.sample(BATCH_SIZE)
            # 构造数据集
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
                'dones': d,
            }
            # 模型训练
            dqn_agent.learn(transition_dict)

        # 检查是否有设备发完数据
        env.check_empty()

        # 保存模型
        if TO_SAVE_MODEL and episode == EPISODE_LOOP - 1 and satellite_moment == 0:
            torch.save(dqn_agent.eval_net.state_dict(),
                       'E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\dqn_' + 'eval.pth')
            torch.save(dqn_agent.target_net.state_dict(),
                       'E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\dqn_' + 'target.pth')
            print('DQN模型已保存')


# 用DDPG网络决策的
def satellite_update_ddpg():
    # time of communication between satellite and ground equipment
    for satellite_moment in range(SATELLITE_TIME):
        # 检查地面设备是否全部发送完数据
        print('**第' + str(episode) + '个周期', '，第' + str(satellite_moment) + '个卫星时刻')
        if env.check_ending(satellite_moment):
            break

        # 地面设备发送信息帧至卫星
        tf_maps, receive_information = env.equipment_send_sonframe()

        # 卫星侧接收到信号之后对时频图进行分析，对信息帧位置进行优化，以达到最大的利用率
        # ------------------------------------- #
        # 深度强化学习DDPG方法
        # ------------------------------------- #
        state_row1 = np.array([sum(row) for row in tf_maps])
        state_row2 = np.array(TIME_SLOT - state_row1)
        state = np.concatenate((state_row1, state_row2), 0)

        # 计算随机接入效率
        env.cal_reff(sum(state_row1))
        # 获取当前状态对应的动作
        action_weights = ddpg_agent.take_action_weights(state) # 动作权重
        sigma = max(0.01, 1.0 - episode / (EPISODE_LOOP / 200) * 0.01)  # 每个循环减小噪声强度
        noisy_weights = ddpg_agent.add_noise_to_weights(action_weights, sigma=sigma)
        actions = env.transfer_action(noisy_weights)

        # 环境更新
        next_state, reward, done = env.update_step(state_row2, actions)

        # 更新经验回放池
        ddpg_replay_buffer.add(state, action_weights, reward, next_state, done)

        # 如果经验池超过容量，开始训练
        if ddpg_replay_buffer.size() > int(BUFFER_SIZE/5):
            # 经验池随机采样batch_size组
            s, a, r, ns, d = ddpg_replay_buffer.sample(BATCH_SIZE)
            # 构造数据集
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
                'dones': d,
            }
            # 模型训练
            ddpg_agent.update(transition_dict)
        # END --------------------------------- #

        # 检查是否有设备发完数据
        env.check_empty()

        # 保存模型
        if TO_SAVE_MODEL and episode == EPISODE_LOOP - 1 and satellite_moment == 0:
            torch.save(ddpg_agent.actor.state_dict(),
                       'E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\ddpg_' + 'actor.pth')
            torch.save(ddpg_agent.critic.state_dict(),
                       'E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\ddpg_' + 'critic.pth')
            torch.save(ddpg_agent.target_actor.state_dict(),
                       'E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\ddpg_' + 'target_actor.pth')
            torch.save(ddpg_agent.target_critic.state_dict(),
                       'E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\ddpg_' + 'target_critic.pth')
            print('DDPG模型已保存')

if __name__ == "__main__":
    # DDPG经验回放池实例化
    ddpg_replay_buffer = DDPG_ReplayBuffer(capacity=BUFFER_SIZE)
    # DQN经验回放池实例化
    dqn_replay_buffer = DQN_ReplayBuffer(capacity=BUFFER_SIZE)
    # DDPG模型实例化
    ddpg_agent = DDPG(n_states=2 * ACTIONS_SPACE,  # 状态数
                 n_hiddens=256,  # 隐含层数
                 n_actions=ACTIONS_SPACE,  # 动作数
                 actor_lr=0.001,  # 策略网络学习率
                 critic_lr=0.001,  # 价值网络学习率
                 tau=0.001,  # 软更新系数
                 gamma=0.99,  # 折扣因子
                 device=device
                 )
    # DQN模型实例化
    dqn_agent = DQN(n_states=2 * ACTIONS_SPACE, # 状态数
                    n_hiddens=256,  # 隐含层数
                    n_actions=ACTIONS_SPACE,  # 动作数
                    lr=0.001, # 学习率
                    tau=0.001,  # 软更新系数
                    gamma=0.99,  # 折扣因子
                    device=device
                    )
    # 参数类实例化
    sharedata = SharedData()
    sharedata_reward = sharedata.reward_queue
    sharedata_reff = sharedata.reff_queue
    sharedata_channel = sharedata.channel_num
    sharedata_devices = sharedata.sucdevices_cou

    # 执行训练
    for episode in range(EPISODE_LOOP):
        env = SatelliteIoT(sate_time=SATELLITE_TIME, # 卫星的覆盖时间
                           actions_space=ACTIONS_SPACE, # 卫星可以分配的不同频点子信道数
                           frame_length=FRAME_LENGTH, # 地面设备可选的总帧长
                           time_slot=TIME_SLOT, # 每个信道的时隙块数
                           pre_training=IS_PRE_TRAINING, # 是否是预训练
                           equipment_alpha=EQUIPMENT_ALPHA, # 地面设备数量折扣因子
                           sharedata_channel=sharedata_channel,
                           sharedata_reward=sharedata_reward, # 奖励记录队列
                           sharedata_reff=sharedata_reff, # 平均随机接入效率记录队列
                           sharedata_devices=sharedata_devices # 成功完成传输的设备数量
                           )
        if IS_DDPG:
            satellite_update_ddpg()
        else:
            satellite_update_dqn()

    # 绘制性能图
    ddpg_agent.plot_actor_cost()
    # sharedata.plot_reward_cost()
    # sharedata.plot_reff()
