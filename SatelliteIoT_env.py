import numpy as np
from scipy import stats
import math
import pandas as pd
import matplotlib.pyplot as plt

class SharedData:
    def __init__(self):
        self.channel_num = [0]
        self.reward_queue = [0] # 奖励队列
        self.reff_queue = [] # 随机接入效率队列
        self.sucdevices_cou = [] # 成功完成传输的设备数量

    # 绘制损失函数
    def plot_reward_cost(self):
        plt.plot(np.arange(len(self.reward_queue)), self.reward_queue)
        plt.xlabel("step")
        plt.ylabel("reward")
        plt.title("reward vs time")
        plt.show()

    # 绘制随机接入效率
    def plot_reff(self):
        plt.plot(np.arange(len(self.reff_queue)), self.reff_queue)
        # 保存reff_queue
        df = pd.DataFrame(self.reff_queue)
        df.to_csv(r'E:\PYTHON\RandomAccess_DeepLearn\Data_toorigin\reff_queue_500_500.csv', index=False)
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.xlabel("Iteration")
        plt.ylabel("ARAE")
        plt.show()

    def plot_sucdevices_cou(self):
        plt.plot(np.arange(len(self.sucdevices_cou)), self.sucdevices_cou)
        # 保存sucdevices_cou
        df = pd.DataFrame(self.sucdevices_cou)
        df.to_csv(r'E:\PYTHON\RandomAccess_DeepLearn\Data_toorigin\bandit_sucdevices_havematchRA_10_1000.csv', index=False)
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.xlabel("Iteration")
        plt.ylabel("sur_devices")
        plt.show()


class SatelliteIoT:
    def __init__(self, sate_time, actions_space, frame_length, time_slot, pre_training, equipment_alpha,
                 sharedata_channel, sharedata_reward, sharedata_reff, sharedata_devices):
        self.T_time = sate_time  # 卫星的覆盖时间
        self.actions_space = actions_space # 卫星可以分配的不同频点子信道数
        self.time_slot = time_slot # 每个信道的时隙块数
        self.frame_length = frame_length # 地面设备可选的总帧长
        self.pre_training = pre_training # 表示是否是预训练

        # 生成卫星整个覆盖时间内要进行随机接入的设备数量（分为时效性业务设备和非时效性业务设备）
        # 时效性设备服从Beta分布
        self.ts_count = stats.poisson.rvs(mu=1500, size=1).item() * equipment_alpha
        # 非时效性设备服从泊松分布
        self.nts_count = stats.poisson.rvs(mu=1500, size=1).item() * equipment_alpha

        # 每个设备随机挑选一个总帧长
        self.ts_frame = np.array(np.random.choice(self.frame_length, size = self.ts_count))
        self.nts_frame = np.array(np.random.choice(self.frame_length, size = self.nts_count))
        # 计算每个设备的子帧长
        if self.pre_training:
            self.ts_sonframe = self.ts_frame
            self.nts_sonframe = self.nts_frame
        else:
            self.ts_sonframe = np.array(self.ts_frame/10, dtype = int)
            self.nts_sonframe = np.array(self.nts_frame/10, dtype = int)

        # 创建一些指示数组，用来指示当前状态
        self.ts_done = np.zeros([1, 0], dtype=int) # 表示已完成传输的ts设备
        self.nts_done = np.zeros([1, 0], dtype=int) # 表示已完成传输的nts设备
        self.ts_todo = np.array([np.array([x for x in range(self.ts_count)]), self.ts_frame, # 表示待传输的ts设备。第一行表示设备的索引
                                 self.ts_sonframe, np.array([1]*self.ts_count)]) # 第二行表示对应设备的总帧长，第三行表示对应设备的子帧长，第四行表示设备类型（ts=1）
        self.nts_todo = np.array([np.array([y for y in range(self.nts_count)]), self.nts_frame, # 表示待传输的nts设备。第一行表示设备的索引
                                  self.nts_sonframe, np.array([0]*self.nts_count)]) # 第二行表示对应设备的总帧长，第三行表示对应设备的子帧长，第四行表示设备类型（nts=0）
        self.equipment_doing = np.zeros([5, 0], dtype=int) # 正在传输的设备.第一行表示设备的索引，第二行表示设备剩余帧长，第三行表示设备子帧长，第四行表示设备类型，第五行表示所在的子信道序号
        self.time_frequency_slot = [[] for _ in range(self.actions_space)] # 时频图，每一行代表一个子信道（因为每个信道不规则，所以使用list）
        
        # 用于画图，观察性能的参数
        self.sharedata_channel = sharedata_channel
        self.sharedata_reward = sharedata_reward
        self.sharedata_reff = sharedata_reff
        self.sharedata_devices = sharedata_devices

    # 得到时效性业务设备和非时效性业务设备的数量
    def get_equipment_count(self):
        return self.ts_count+self.nts_count, self.ts_count, self.nts_count

    # 检查是否有设备发完数据了
    def check_empty(self):
        idx = 0
        while idx < np.size(self.equipment_doing, 1):
            if self.equipment_doing[1, idx] == 0:
                if self.equipment_doing[3, idx] == 1:
                    self.ts_done = np.append(self.ts_done, [self.equipment_doing[0, idx]])
                else:
                    self.nts_done = np.append(self.nts_done, [self.equipment_doing[0, idx]])
                self.equipment_doing = np.delete(self.equipment_doing, idx, axis=1)
            else:
                idx += 1
        self.sharedata_devices.append(self.ts_done.shape[0] + self.nts_done.shape[0]) # 记录当前成功传输了多少设备
        self.update_map()

    # 进行设备匹配，找到最合适的设备
    def equipment_to_emptyslot(self):
        fit_ts = False
        is_match = False
        if len(self.time_frequency_slot) != 0:
            sum_subset = [sum(row) for row in self.time_frequency_slot] # 计算时频图的每个子信道已被占用的时隙块
            for idx in range(len(sum_subset)):
                if sum_subset[idx] != 0 and sum_subset[idx] != self.time_slot:
                    is_match = True
                    useful_slot = self.time_slot - sum_subset[idx] # 代表此信道剩余几个时隙块
                    if np.size(self.ts_todo, 1) != 0: # 首先匹配ts设备
                        element1 = self.fit_length('ts', useful_slot)
                        if element1.size != 0:
                            choose = np.squeeze(element1[0])
                            add_col = np.concatenate((np.reshape([self.ts_todo[:, choose]], (4, 1)), np.array([[idx]])))
                            self.equipment_doing = np.append(self.equipment_doing, add_col, axis=1)  # 把要传输的设备的索引加入列表中
                            self.ts_todo = np.delete(self.ts_todo, choose, axis=1)
                            fit_ts = True
                    if np.size(self.nts_todo, 1) != 0 and not fit_ts: # 如果没有匹配到ts设备，那就匹配nts设备
                        element2 = self.fit_length('nts', useful_slot)
                        if element2.size != 0:  # 有匹配的nts设备
                            choose = np.squeeze(element2[0])
                            add_col = np.concatenate((np.reshape([self.nts_todo[:, choose]], (4, 1)), np.array([[idx]])))
                            self.equipment_doing = np.append(self.equipment_doing, add_col, axis=1)  # 把要传输的设备的索引加入列表中
                            self.nts_todo = np.delete(self.nts_todo, choose, axis=1)
        return is_match

    # 用于“equipment_to_emptyslot”函数
    def fit_length(self, euipment_type, useful_slot):
        slot_len = useful_slot
        useful_equipment = np.empty((1, 1), dtype=int)
        while slot_len > 0:
            if euipment_type == 'ts':
                element = np.argwhere(self.ts_todo[2, :] == slot_len)
            else:
                element = np.argwhere(self.nts_todo[2, :] == slot_len)
            useful_equipment = np.vstack((useful_equipment, element))
            slot_len -= 1
        useful_equipment = np.delete(useful_equipment, 0, 0) # 删除第一个没用的数据
        return useful_equipment

    # 检查地面设备是否全部传输完成
    def check_ending(self, satellite_moment):
        if satellite_moment < self.T_time:
            if np.size(self.ts_todo, 1) == 0 and np.size(self.nts_todo, 1) == 0:
                print("【在第" + str(satellite_moment) + "个时刻，" + "所有地面设备全部传输完成】" + '\n' +
                      "延迟敏感设备总个数为：" + str(self.ts_count) + "，延迟不敏感设备总个数为：" + str(self.nts_count))
                return True
            else:
                return False
        elif satellite_moment == self.T_time:
            if np.size(self.ts_todo, 1) == 0 and np.size(self.nts_todo, 1) == 0:
                print("【在卫星照射结束时，所有地面设备全部传输完成】" + '\n' +
                      "延迟敏感设备总个数为：" + str(self.ts_count) + "，延迟不敏感设备总个数为：" + str(self.nts_count))
            elif np.size(self.ts_todo, 1) == 0 and np.size(self.nts_todo, 1) != 0:
                print("【在卫星照射结束时，延迟敏感设备全部传输完成，延迟不敏感设备未全部传输完成】" + '\n' +
                      "延迟敏感设备总个数为：" + str(self.ts_count) + "，延迟不敏感设备总个数为：" + str(self.nts_count))
            elif np.size(self.ts_todo, 1) != 0 and np.size(self.nts_todo, 1) == 0:
                print("【在卫星照射结束时，延迟敏感设备未全部传输完成，延迟不敏感设备全部传输完成】" + '\n' +
                      "延迟敏感设备总个数为：" + str(self.ts_count) + "，延迟不敏感设备总个数为：" + str(self.nts_count))
            else:
                print("【在卫星照射结束时，延迟敏感设备和延迟不敏感设备都未全部传输完成】" + '\n' +
                      "延迟敏感设备总个数为：" + str(self.ts_count) + "，延迟不敏感设备总个数为：" + str(self.nts_count))
            return True

    # 日志汇报
    def log_report(self):
        pass

    # -----------------------------------------------------------------------------
    # NOTE: The remaining components are temporarily restricted due to
    # ongoing laboratory project confidentiality and will be published in a
    # future update after project completion and confidentiality clearance.
    # -----------------------------------------------------------------------------