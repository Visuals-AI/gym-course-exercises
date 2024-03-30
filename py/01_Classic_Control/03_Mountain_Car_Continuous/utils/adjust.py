#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------


def adjust(next_obs, reward, min_x, max_x):
    '''
    奖励重塑，包括对停滞的惩罚
    :params: next_obs 执行当前 action 后、小车处于的状态
    :params: reward 执行当前 action 后、小车获得的奖励
    :params: min_x 本回合训练中、小车走得离目标地点最远的距离（为了借力）
    :params: max_x 本回合训练中、小车走得离目标地点最近的距离
    :return: 重塑后的 (reward, min_x, max_x)
    '''
    x = next_obs[0][0]     # 小车位置
    speed = next_obs[0][1] # 小车速度，向前为正、向后为负

    # 鼓励小车向目标移动并更新最远/最近位置
    if x > max_x:
        reward += (x - max_x) * 10  # 刷新距离目标最近的距离，给予最大奖励
        max_x = x
    elif x < min_x:
        reward += (min_x - x) * 5  # 刷新距离目标最远的距离，也给予一定的奖励（因为可以借力）
        min_x = x

    # 根据速度给予奖励或惩罚
    if abs(speed) > 0.01:  # 如果小车有明显的移动
        reward += abs(speed) * 5  # 根据速度的绝对值给予奖励
    else:
        reward -= 1  # 对于几乎没有移动的情况给予小的惩罚

    return (reward, min_x, max_x)
