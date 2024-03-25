#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------

import numpy as np
from conf.settings import MAX_STEP
from utils.rotation import RotationDetector
from tools.utils import is_close_to_zero


def adjust(obs, action, reward, rotation: RotationDetector, step):
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

    :params: obs 当前智能体处于的状态
    :params: action 当前智能体准备执行的动作
    :params: reward 执行当前 action 后、智能体当前步获得的奖励
    :params: rotation 旋转记录器
    :params: step 当前步数
    :return: 重塑后的 (reward, min_x, max_x)
    '''
    terminated = False
    x = obs[0][0]   # 自由端的 x 坐标
    y = obs[0][1]   # 自由端的 y 坐标
    v = obs[0][2]   # 自由端的角速度，顺时针为- 逆时针为+ 
    a = action[0]   # 对自由端施加的扭矩，顺时针为- 逆时针为+ 
    # print(f"reward: {reward}, x: {x}, y: {y}, v: {v}, a: {a}")

    # 若发生了至少一圈旋转，给予最大惩罚（强制终止）
    is_rotation = rotation.update(x, y, v, a)
    if is_rotation :
        reward = -1000
        terminated = True
        return (reward, terminated)
    

    # 理想状态 x=1 y=0 theat=0 v=0 
    # Gym 公式的系数的大小说明了: 角度 > 角速度 > 动作值
    # 根据重要程度重新设计奖励函数

    thresholds_xy = [i/100 for i in range(1, 50)]    # xy 阈值等级，从 0.01 -> 0.5  [ 0.01, 0.02, .., 0.5]
    reward_xy = [50 - i for i in range(0, 50)]       # xy 阈值等级奖励，从 50 -> 1


    if x < 0 :
        t_xy = reversed(np.arange(-1, 0.01, 0.01))
        r_xy = [10 + i * 10 for i in t_xy]
        print(t_xy)
        print(r_xy)
        for idx, threshold in enumerate(t_xy) :
            if is_close_to_zero(1 + x, threshold) and is_close_to_zero(y, threshold) :
                reward += r_xy[idx]
                break

    else :
        t_xy = reversed(np.arange(0, 1.01, 0.01))
        r_xy = [20 + i * 10 for i in t_xy]

        # 根据摆锤接近垂直位置的距离，给予不同的等级的奖励
        # 因为影响奖励系数公式最大的因素是角度，故转换为 xy 坐标距离
        for idx, threshold in enumerate(t_xy) :
            if is_close_to_zero(1 - x, threshold) and is_close_to_zero(y, threshold) :
                reward += r_xy[idx]
                # print(f"xy reward: {reward}, x: {x}, y: {y}, v: {v}")
                reward += _adjust(reward, v)
                break
    return (reward, terminated)


# 根据摆锤在接近垂直位置距离的同时、检查角速度
# 角速度越接近 0，给予更高的追加奖励，因为速度接近 0 等价于不易离开垂直位置
def _adjust(reward, angular_velocity) :
    thresholds_v = [i/10 for i in range(1, 21)]     # 角速度 阈值等级，从 0.1 -> 2
    reward_v = [20 - 10*i for i in range(0, 20)]    # 角速度 阈值等级奖励，从 20 -> 1
    
    t_v = reversed(np.arange(0, 2.01, 0.01))
    r_v = [10 + i * 10 for i in t_v]
    for idx, threshold in enumerate(t_v) :
        if is_close_to_zero(angular_velocity, threshold) :
            reward += r_v[idx]
            # print(f"v reward: {reward}, v: {angular_velocity}")
            break
    return reward

