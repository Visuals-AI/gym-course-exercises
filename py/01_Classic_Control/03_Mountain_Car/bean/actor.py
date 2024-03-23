#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/17 20:33
# -----------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义 Actor 网络
# 其实和前面 DQN 的定义基本是一样的，区别只有两处
class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_size, 400)     # 区别一： 连接层的神经元数量不一样
        self.layer2 = nn.Linear(400, 300)            # 神经元数量越多、学习复杂问题的能力越好，但是需要更多资源
        self.layer3 = nn.Linear(300, action_size)    # 前面 DQN 每层是 24 个神经元、这里是 400 个，根据先验经验得到的，可调整 
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))    # 却别二： 离散 -> 连续
                                                            # 这步很关键，通过 tan 函数把离散值归一化，落在 [-1, 1] 的标准区间
        return x                                            # 然后通过 * max_action 把标准区间映射回去真正的动作区间

