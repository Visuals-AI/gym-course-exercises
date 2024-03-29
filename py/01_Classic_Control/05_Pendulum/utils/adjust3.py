#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------

from utils.terminate import TerminateDetector
from tools.utils import is_close_to_zero


# 三四象限的坐标阶段阈值和奖励
T34_XY = [ i/100 for i in range(1, 101) ]
R34_XY = [ i * 10 - 10 for i in T34_XY ]


# 一二象限的坐标阶段阈值和奖励
T12_XY = [ i/100 for i in range(0, 101) ]
R12_XY = [ 20 - i * 10 for i in T12_XY ]


# 一二象限的角速度阶段阈值和奖励
T12_V = [ i/100 for i in range(0, 201) ]
R12_V = [ 30 - i * 10 for i in T12_V ]

 

def adjust(obs, action, reward, td: TerminateDetector, step):
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
    
    根据 gym 的奖励公式： r = -(theta^2 + 0.1 * theta_dt^2 + 0.001 * torque^2)
    所以 reward 由角度的平方、角速度的平方以及动作值决定。
    当智能体越往上，角度越大，reward 越大，从而鼓励智能体往上摆；
    角速度越大，reward 越大，从而鼓励智能体快速往上摆；
    动作值越大，摆动的速度也就越大，从而加速智能体往上摆。
    
    系数的大小说明了: 角度 > 角速度 > 动作值。
    因为摆过头，角度 theta 就下降得快，reward 就跌得快。

    :params: obs 当前智能体处于的状态
    :params: action 当前智能体准备执行的动作
    :params: reward 执行当前 action 后、智能体当前步获得的奖励
    :params: td 终止计算器
    :params: step 当前步数
    :return: 重塑后的 (reward, min_x, max_x)
    '''
    terminated = False
    x = obs[0][0]   # 自由端的 x 坐标
    y = obs[0][1]   # 自由端的 y 坐标
    v = obs[0][2]   # 自由端的角速度，顺时针为- 逆时针为+ 
    a = action[0]   # 对自由端施加的扭矩，顺时针为- 逆时针为+ 

    # 判断是否触发终止条件
    td.update(x, y, v, a)
    if td.is_terminate() :
        reward = -1000
        terminated = True
        return (reward, terminated)
    

    # 理想状态 x=1 y=0 theat=0 v=0 
    # Gym 公式的系数的大小说明了: 角度 > 角速度 > 动作值
    # 根据重要程度重新设计奖励函数：
    #   根据摆锤接近垂直位置的距离，给予不同的等级的奖励
    #   因为影响奖励系数公式最大的因素是角度，故转换为 xy 坐标距离

    reward = 0

    if x < 0 :

        # x 接近 -1， y 接近 0，惩罚越重
        for idx, threshold in enumerate(T34_XY) :
            if is_close_to_zero(1 + x, threshold) and is_close_to_zero(y, threshold) :
                reward += R34_XY[idx]
                # print(f"xy reward: {reward}, x: {x}, y: {y}, v: {v}, step: {step}")
                break
    else :

        # x 接近 1， y 接近 0，奖励越大
        for idx, threshold in enumerate(T12_XY) :
            if is_close_to_zero(1 - x, threshold) and is_close_to_zero(y, threshold) :
                reward += R12_XY[idx]
                # print(f"xy reward: {reward}, x: {x}, y: {y}, v: {v}, step: {step}")
                reward += _adjust(reward, v)
                break

        # 完全在垂直正中且速度为 0，给予最大奖励
        if is_close_to_zero(1 - x) and is_close_to_zero(y) and is_close_to_zero(v) :
            reward += 1000

    return (reward, terminated)


# 根据摆锤在接近垂直位置距离的同时、检查角速度
# 角速度越接近 0，给予更高的追加奖励，因为速度接近 0 等价于不易离开垂直位置
def _adjust(reward, angular_velocity) :
    for idx, threshold in enumerate(T12_V) :
        if is_close_to_zero(angular_velocity, threshold) :
            reward += R12_V[idx]
            # print(f"v reward: {reward}, v: {angular_velocity}")
            break
    return reward

