#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义 Critic 网络（Q 网络，类比 DQN 中的 Q 值、在连续问题中升维成网络了）
class Critic(nn.Module):

    # 这两个网络结构完全相同，但是它们各自独立地学习和更新，以便为同一状态-动作对提供两个略有不同的价值估计
    # 这样做可以：
    #   1. 减少过度估计：通过采用两个Critic网络的最小值作为目标Q值，TD3算法减少了Q值的过度估计倾向。过度估计会导致策略评估不准确，从而影响学习性能。
    #   2. 提高稳定性：两个独立的Critic网络通过提供两个独立的价值估计，增加了学习过程的稳定性。这有助于防止算法过于依赖于某个可能不准确的单一价值估计。
    def __init__(self, obs_size, act_size):
        super(Critic, self).__init__()

        # 第一个 Critic 网络
        self.layer1 = nn.Linear(obs_size + act_size, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

        # 第二个 Critic 网络
        self.layer4 = nn.Linear(obs_size + act_size, 400)
        self.layer5 = nn.Linear(400, 300)
        self.layer6 = nn.Linear(300, 1)


    # x: obs_batch 批量的状态张量，形状为 [batch_size, obs_size]， 如 32x2
    # u: act_batch 批量的动作张量，形状为 [batch_size, act_size]， 如 32x1
    # “批量”指从【经验回放存储】中每次所取出用于 TD3 学习的经验数量
    def forward(self, x, u) :

        # 表示把两个张量的第 1 维拼接在一起形成一个新的张量
        # 即 xu = [ obs_size + act_size ]
        xu = torch.cat([x, u], 1)

        # 第一个 Critic 网络的前向传播
        x1 = F.relu(self.layer1(xu))
        x1 = F.relu(self.layer2(x1))
        q1 = self.layer3(x1)

        # 第二个 Critic 网络的前向传播
        x2 = F.relu(self.layer4(xu))
        x2 = F.relu(self.layer5(x2))
        q2 = self.layer6(x2)
        return q1, q2


    # 使用 Q1 网络更新 Actor 的 Q 值估计
    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer1(xu))
        x1 = F.relu(self.layer2(x1))
        q1 = self.layer3(x1)
        return q1
