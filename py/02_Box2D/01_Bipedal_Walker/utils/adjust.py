#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------

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

    return (reward, terminated)

