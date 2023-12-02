#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------

import torch
import torch.nn as nn


# 神经网络模型： 简单的三层全连接网络
# 可以根据具体问题和实验结果进行调整层数、每层的神经元数量等
class DQN(nn.Module):   # 在 PyTorch 中，nn.Module 是所有神经网络模块的基类

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)    # 定义第一个全连接层（fc1），它接收状态输入，并将其映射到24个隐藏单元
        self.fc2 = nn.Linear(24, 24)            # 定义第二个全连接层（fc2），它进一步处理来自第一层的数据。
        self.fc3 = nn.Linear(24, action_size)   # 定义第三个全连接层（fc3），它将隐藏层的输出映射到动作空间的大小，即为每个可能的动作生成一个 Q 值


    # 覆盖 nn.Module 中的 forward 方法（前向传播函数）
    # 在 PyTorch 中，只需定义前向传播函数，而后向传播函数（用于计算梯度）是由PyTorch的autograd自动定义的
    def forward(self, x):
        x = torch.relu(self.fc1(x))     # 应用了第一个全连接层，并对其输出应用了ReLU（修正线性单元）激活函数。ReLU激活函数用于增加网络的非线性，使其能够学习更复杂的函数。
        x = torch.relu(self.fc2(x))     # 对第二个全连接层的输出应用了ReLU激活
        return self.fc3(x)              # 网络返回第三个全连接层的输出。这个输出代表了在给定状态下每个动作的预期Q值


