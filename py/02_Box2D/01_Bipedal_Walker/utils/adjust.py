#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------

# https://aijishu.com/a/1060000000162444
from utils.terminate import TerminateDetector


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

    r1_torque = action[0]       # 右腿关节1的扭矩：连接身体和大腿的关节的旋转力矩，顺时针为负、逆时针为正
    r2_torque = action[1]       # 右腿关节2的扭矩：连接大腿和小腿的关节的旋转力矩，顺时针为负、逆时针为正
    l1_torque = action[2]       # 左腿关节1的扭矩：连接身体和大腿的关节的旋转力矩，顺时针为负、逆时针为正
    l2_torque = action[3]       # 左腿关节2的扭矩：连接大腿和小腿的关节的旋转力矩，顺时针为负、逆时针为正

    h_v = obs[0][0]                # 机器人主体的水平速度：沿 x 轴的速度
    v_v = obs[0][1]                # 机器人主体的垂直速度：沿 y 轴的速度
    a_v = obs[0][2]                # 机器人主体的角速度：主体围绕中心点的旋转速率，顺时针为负、逆时针为正
    body_angle = obs[0][3]         # 机器人主体的角度： 与水平面 x 轴的夹角
    r1_angle = obs[0][4]           # 右腿关节1的角度： 大腿与水平面 x 轴的夹角
    r1_a_v = obs[0][5]             # 右腿关节1的角速度： 顺时针为负、逆时针为正
    r1_landed = obs[0][6]          # 右腿关节1是否已触碰地面： 即身体中心已着地，1表示接触，0表示未接触
    r2_angle = obs[0][7]           # 右腿关节2的角度： 大腿与水平面 x 轴的夹角
    r2_a_v = obs[0][8]             # 右腿关节2的角速度： 顺时针为负、逆时针为正
    r2_landed = obs[0][9]          # 右腿关节2是否已触碰地面： 即膝盖已着地，1表示接触，0表示未接触
    l1_angle = obs[0][10]          # 左腿关节1的角度： 大腿与水平面 x 轴的夹角
    l1_a_v = obs[0][11]            # 左腿关节1的角速度： 顺时针为负、逆时针为正
    l1_landed = obs[0][12]         # 左腿关节1是否已触碰地面： 即身体中心已着地，1表示接触，0表示未接触
    l2_angle = obs[0][13]          # 左腿关节2的角度： 大腿与水平面 x 轴的夹角
    l2_a_v = obs[0][14]            # 左腿关节2的角速度： 顺时针为负、逆时针为正
    l2_landed = obs[0][15]         # 左腿关节2是否已触碰地面： 即膝盖已着地，1表示接触，0表示未接触
    h_d = obs[0][16]               # 机器人身体中心点 距离前方第一个被感知的 地形位置（即 obs[0][18]）的水平距离
    th1 = obs[0][17]               # 机器人感知到前方 7 个连续点的第 1 个的地形高度（y 轴）
    th2 = obs[0][18]               # 机器人感知到前方 7 个连续点的第 2 个的地形高度（y 轴）
    th3 = obs[0][19]               # 机器人感知到前方 7 个连续点的第 3 个的地形高度（y 轴）
    th4 = obs[0][20]               # 机器人感知到前方 7 个连续点的第 4 个的地形高度（y 轴）
    th5 = obs[0][21]               # 机器人感知到前方 7 个连续点的第 5 个的地形高度（y 轴）
    th6 = obs[0][22]               # 机器人感知到前方 7 个连续点的第 6 个的地形高度（y 轴）
    th7 = obs[0][23]               # 机器人感知到前方 7 个连续点的第 7 个的地形高度（y 轴）
    # 状态索引不对

    # 扣分项：
    # 1. 膝盖着地
    # 2. 屁股着地

    # https://blog.csdn.net/weixin_48370148/article/details/114549032
    if reward == -100:
        reward = -1
    reward = reward * 10

    return (reward, terminated)

