import numpy as np
import pandas as pd
import torch
import math
import matplotlib.pyplot as plt
from SatelliteIoT_env import SatelliteIoT, SharedData
from PositionOptimization_DDPG import DDPG
from PositionOptimization_DQN import DQN
import PositionOptimization_DP as DP_plan
import PositionOptimization_Bandit as Bandit_plan
import time

## 场景说明
# 其中最关键的点是：对地面设备进行访问控制（基于匹配的）和资源分配（基于Lyapunov优化的）
## END
## 程序说明
#
## END

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 超参数
SATELLITE_TIME = 100 # 每个卫星覆盖的时间
ACTIONS_SPACE = 10 # 卫星可以分配的不同频点子信道数
TIME_SLOT = 100 # 每个信道的时隙块数
FRAME_LENGTH = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) # 地面设备可选的总帧长
EQUIPMENT_ALPHA = 100 # 地面设备数量折扣因子

run_time_50_200 = [] # DDPG运行时间记录
signalling_overhead = [0] # 信令开销记录

def satellite_update():
    # time of communication between satellite and ground equipment
    for satellite_moment in range(SATELLITE_TIME):
        print('** 第' + str(satellite_moment) + '个卫星时刻')
        # 检查地面设备是否全部发送完数据
        if env.check_ending(satellite_moment):
            break

        # 地面设备发送信息帧至卫星
        tf_maps, receive_information = env.equipment_send_sonframe()

        start_time = time.time()
        # 卫星侧接收到信号之后对时频图进行分析，对信息帧位置进行优化，以达到最大的利用率
        # ------------------------------------- #
        # 深度强化学习方法
        # ------------------------------------- #
        state_row1 = np.array([sum(row) for row in tf_maps])
        state_row2 = np.array(TIME_SLOT - state_row1)
        state = np.concatenate((state_row1, state_row2), 0)

        # 计算随机接入效率
        # env.cal_reff(sum(state_row1))

        # DDPG或DQN获取当前状态对应的动作
        # action_weights = dqn_agent.choose_action(state)  # 使用DQN
        action_weights = ddpg_agent.take_action_weights(state) # 使用DDPG
        actions = env.transfer_action(action_weights) # 根据权重得到动作

        # ------------------------------------- #
        # 动态规划方法
        # ------------------------------------- #
        dp_actions = np.array(DP_plan.dp_agent(state_row1, max_sum=TIME_SLOT))
        # END --------------------------------- #

        # ------------------------------------- #
        # Bandit方法
        # ------------------------------------- #
        bandit_actions = np.array(Bandit_plan.bandit_partition_approx(state_row1, max_sum=TIME_SLOT, pool_size=15, seed=42, max_rounds=100))
        # END --------------------------------- #

        # 环境更新
        env.update_step(state_row2, bandit_actions)
        # END --------------------------------- #

        end_time = time.time()

        # 检查是否有设备发完数据
        env.check_empty()

        # 对于零碎时隙，进行设备匹配
        is_match = env.equipment_to_emptyslot()
        if is_match:
            signalling_overhead.append(signalling_overhead[satellite_moment] + 1)
        else:
            signalling_overhead.append(signalling_overhead[satellite_moment])

        run_time_50_200.append(end_time - start_time)



if __name__ == "__main__":
    # 参数类实例化
    sharedata = SharedData()
    sharedata_reward = sharedata.reward_queue
    sharedata_reff = sharedata.reff_queue
    sharedata_channel = sharedata.channel_num
    sharedata_devices = sharedata.sucdevices_cou

    env = SatelliteIoT(sate_time=SATELLITE_TIME,  # 卫星的覆盖时间
                       actions_space=ACTIONS_SPACE,  # 卫星可以分配的不同频点子信道数
                       frame_length=FRAME_LENGTH,  # 地面设备可选的总帧长
                       time_slot=TIME_SLOT,  # 每个信道的时隙块数
                       pre_training=False,  # 是否是预训练
                       equipment_alpha=EQUIPMENT_ALPHA,  # 地面设备数量折扣因子
                       sharedata_channel=sharedata_channel,
                       sharedata_reward=sharedata_reward,  # 奖励记录队列
                       sharedata_reff=sharedata_reff,  # 平均随机接入效率记录队列
                       sharedata_devices=sharedata_devices  # 成功完成传输的设备数量
                       )
    # 模型实例化
    ddpg_agent = DDPG(n_states=2*ACTIONS_SPACE,  # 状态数
                 n_hiddens=256,  # 隐含层数
                 n_actions=ACTIONS_SPACE,  # 动作数
                 actor_lr=0.001,  # 策略网络学习率
                 critic_lr=0.001,  # 价值网络学习率
                 tau=0.001,  # 软更新系数
                 gamma=0.99,  # 折扣因子
                 device=device
                 )
    # DQN模型实例化
    dqn_agent = DQN(n_states=2 * ACTIONS_SPACE,  # 状态数
                    n_hiddens=256,  # 隐含层数
                    n_actions=ACTIONS_SPACE,  # 动作数
                    lr=0.001,  # 学习率
                    tau=0.001,  # 软更新系数
                    gamma=0.99,  # 折扣因子
                    device=device
                    )
    # 把预训练好的DDPG模型加载进来
    ddpg_agent.actor.load_state_dict(
        torch.load('E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\ddpg_' + 'actor.pth'))
    ddpg_agent.critic.load_state_dict(
        torch.load('E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\ddpg_' + 'critic.pth'))
    ddpg_agent.target_actor.load_state_dict(torch.load(
        'E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\ddpg_' + 'target_actor.pth'))
    ddpg_agent.target_critic.load_state_dict(torch.load(
        'E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\ddpg_' + 'target_critic.pth'))
    dqn_agent.eval_net.load_state_dict(torch.load(
        'E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\dqn_' + 'eval.pth'))
    dqn_agent.target_net.load_state_dict(torch.load(
        'E:\PYTHON\RandomAccess_DeepLearn\RandomAccess_ForPaper\Pre_training_model\dqn_' + 'target.pth'))

    satellite_update()

    sharedata.plot_sucdevices_cou()
    # sharedata.plot_reff()

