#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------

from utils.terminate import TerminateDetector
from tools.utils import is_close_to_zero


#============================
# 1. 全局奖惩（目标方向控制）
#============================
# 1.1. 三四象限的坐标阶段阈值和奖励（越接近垂直向下，惩罚越重）
T34_XY = [ i/100 for i in range(0, 101) ]
R34_XY = [ i * 10 - 30 for i in T34_XY ]

# 1.2. 一二象限的坐标阶段阈值和奖励（越接近垂直向上，奖励越大）
T12_XY = [ i/100 for i in range(0, 101) ]
R12_XY = [ 20 - i * 10 for i in T12_XY ]

# 1.3. 角速度绝对值，避免速度过快刹不住。 小于 5 不奖不罚， 大于 5 惩罚
#      速度计算是有依据的：摆锤从最低点摆动至脱离三四象限，至少需要 4.43 的角速度，所以更大的角速度是没必要的，可以避免摆锤刹不住而转圈
T1234_V = [ i/10 for i in range(50, 81) ]
R1234_V = [ -i for i in T1234_V ]


#============================
# 2. 局部极致奖惩（精准控制）
#============================
# 2.2. 在垂直向上附近 0 - 0.009 范围内微调，越接近 0 奖励越大
TTOP_XY = [ i/1000 for i in range(0, 11) ]
RTOP_XY = [ 100 - i * 10000 for i in TTOP_XY ]

# 2.2. 在垂直向上附近 0 - 0.009 范围内微调，越接近 0 奖励越大
TTOP_V = [ i/1000 for i in range(0, 11) ]
RTOP_V = [ 100 - i * 10000 for i in TTOP_V ]


 

def adjust(obs, action, reward, td: TerminateDetector, step) :
    '''
    奖励重塑。
    :params: obs 当前智能体处于的状态
    :params: action 当前智能体准备执行的动作
    :params: reward 执行当前 action 后、智能体当前步获得的奖励
    :params: td 终止计算器
    :params: step 当前步数
    :return: 重塑后的 (reward, min_x, max_x)
    '''
    terminated = False
    x = obs[0][0]           # 自由端的 x 坐标
    y = obs[0][1]           # 自由端的 y 坐标
    v = obs[0][2]           # 自由端的角速度，顺时针为- 逆时针为+ 
    a = action[0]           # 对自由端施加的扭矩，顺时针为- 逆时针为+ 
    _y = 1 if y >= 0 else -1
    _v = 1 if v >= 0 else -1
    direction = _y * _v     # < 0 向着垂直向上的方向， > 0 向着垂直向下的方向

    # 判断是否触发终止条件
    td.update(x, y, v, a)
    if td.is_terminate() :
        reward = -1000
        terminated = True
        return (reward, terminated)
    

    reward = 0

    # 速度梯度惩罚
    for idx, threshold in enumerate(T1234_V) :
        if is_close_to_zero(v, threshold) :
            reward += R1234_V[idx]
            break

    # 三四象限
    if x < 0 :

        # 距离梯度惩罚
        for idx, threshold in enumerate(T34_XY) :
            if is_close_to_zero(1 + x, threshold) and is_close_to_zero(y, threshold) :
                if direction < 0 :           # 如果方向接近目标，不处罚，奖励归 0
                    reward = 0               #（因为在三四象限需要加速，v 可能需要加速到比较大，故连带取消速度处罚）
                else :
                    reward += R34_XY[idx]    # 如果方向远离目标，处罚
                break

    # 一二象限
    else :

        # 距离梯度奖励（粗粒度距离奖励）
        for idx, threshold in enumerate(T12_XY) :
            if is_close_to_zero(1 - x, threshold) and is_close_to_zero(y, threshold) :
                if direction < 0 :
                    reward += R12_XY[idx]   # 如果方向接近目标，奖励
                    break

        # 距离梯度奖励（细粒度距离奖励，精细控制可以获得更大的追加奖励）
        for idx, threshold in enumerate(TTOP_XY) :
            if is_close_to_zero(1 - x, threshold) and is_close_to_zero(y, threshold) :
                if direction < 0 :
                    reward += RTOP_XY[idx]          # 如果方向精确接近目标，追加奖励
                    reward += _adjust_v(reward, v)  # 如果此时角速度很小，追加奖励
                break
        
        # 完全在垂直正中且速度为 0，给予最大奖励
        if is_close_to_zero(1 - x) and is_close_to_zero(y) and is_close_to_zero(v) :
            reward += 1000

    return (reward, terminated)


# 根据摆锤在接近垂直位置距离的同时、检查角速度
# 角速度越接近 0，等价于不易离开垂直位置，给予更高的追加奖励
def _adjust_v(reward, angular_velocity) :
    for idx, threshold in enumerate(TTOP_V) :
        if is_close_to_zero(angular_velocity, threshold) :
            reward += RTOP_V[idx]
            break
    return reward

