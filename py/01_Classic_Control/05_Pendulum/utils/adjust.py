#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------

from tools.utils import is_close_to_zero


def adjust(next_obs, reward, step):
    '''
    奖励重塑。

    在 Pendulum 问题中，目标是让智能体在到达垂直位置后、能维持更长的步数。

    在 Pendulum 每一步的奖励是根据 theta 的夹角去计算的（非线性）：
      夹角最大，奖励越小，最小奖励为 -16.2736044
      夹角最小，奖励最大，最大奖励为 0

    理想状态下，
        x=1 y=0
        夹角 theta=0  
        角速度 angular_velocity=0 （顺时针为+ 逆时针为-）
    
    根据 gym 的奖励公式： r = -(theta^2 + 0.1 * angular_velocity^2 + 0.001 * action^2)
    所以 reward 由角度的平方、角速度的平方以及动作值决定。
    当智能体越往上，角度越大，reward 越大，从而鼓励智能体往上摆；
    角速度越大，reward 越大，从而鼓励智能体快速往上摆；
    动作值越大，摆动的速度也就越大，从而加速智能体往上摆。
    
    系数的大小说明了: 角度 > 角速度 > 动作值。
    因为摆过头，角度 theta 就下降得快，reward 就跌得快。

    :params: next_obs 执行当前 action 后、智能体处于的状态
    :params: reward 执行当前 action 后、智能体当前步获得的奖励
    :params: step 当前步数
    :return: 重塑后的 (reward, min_x, max_x)
    '''
    x = next_obs[0][0]   # 自由端的 x 坐标
    y = next_obs[0][1]   # 自由端的 y 坐标
    v = next_obs[0][2]   # 自由端的角速度
    # print(f"reward: {reward}, x: {x}, y: {y}, v: {angular_velocity}")

    # 理想状态 x=1 y=0 theat=0  v=0 顺时针为+ 逆时针为- 
    # Gym 公式的系数的大小说明了: 角度 > 角速度 > 动作值
    # 根据重要程度重新设计奖励函数

    thresholds_xy = [i/100 for i in range(1, 11)]       # xy 阈值等级，从 0.01 -> 0.1
    reward_xy = [100 - 10*i for i in range(0, 10)]      # xy 阈值等级奖励，从 100 -> 10

    # 摆锤滞留在三、四象限
    if x < 0 :

        # 第三象限且顺时针向上 或
        # 第四象限且逆时针向上
        if (y > 0 and v > 0) or (y < 0 and v < 0) : 
            reward = 0  # 不与惩罚
        else :
            reward = -step * 0.1

    # 滞留在一、二象限
    else :

        # 第二象限且顺时针向上 或
        # 第一象限且逆时针向上
        if (y > 0 and v > 0) or (y < 0 and v < 0) : 
            reward = 1  # 给予一点奖励
        else :
            reward = -step * 0.1

        # 根据摆锤接近垂直位置的距离，给予不同的等级的奖励
        # 因为影响奖励系数公式最大的因素是角度，故转换为 xy 坐标距离
        for idx, threshold in enumerate(thresholds_xy) :
            if is_close_to_zero(x - 1, threshold) and is_close_to_zero(y, threshold) :
                reward += reward_xy[idx]
                # print(f"xy reward: {reward}, x: {x}, y: {y}, v: {v}")
                reward += _adjust(reward, v)
                break
    return reward


# 根据摆锤在接近垂直位置距离的同时、检查角速度
# 角速度越接近 0，给予更高的追加奖励，因为速度接近 0 等价于不易离开垂直位置
def _adjust(reward, angular_velocity) :
    thresholds_v = [i/10 for i in range(1, 11)]        # 角速度 阈值等级，从 0.1 -> 1
    reward_v = [1000 - 100*i for i in range(0, 10)]    # 角速度 阈值等级奖励，从 1000 -> 100
    
    for idx, threshold in enumerate(thresholds_v) :
        if is_close_to_zero(angular_velocity, threshold) :
            reward += reward_v[idx]
            # print(f"v reward: {reward}, v: {angular_velocity}")
            break
    return reward

