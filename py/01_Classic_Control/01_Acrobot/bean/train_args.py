#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------

import torch.nn as nn
import torch.optim as optim
from bean.dqn import DQN
from collections import deque
from tools.utils import scan_device
from bean.checkpoint import CheckpointManager


class TrainArgs :

    def __init__(self, args, env) -> None:
        # ------------------------------------------
        # 深度 Q 网络（DQN）算法的关键参数和设置
        self.args = args
        self.env = env
        self.cp_mgr = CheckpointManager()        # 自动管理 checkpoint 的记录

        self.obs_size = env.observation_space.shape[0]     # Acrobot 状态空间维度
        self.action_size = env.action_space.n                # Acrobot 动作空间数量

        self.model = DQN(self.obs_size, self.action_size)  # DQN 简单的三层网络模型
        self.memory = deque(maxlen=2000)           # 创建一个双端队列（deque），作为经验回放的存储。当存储超过2000个元素时，最旧的元素将被移除。经验回放是DQN中的一项关键技术，有助于打破经验间的相关性并提高学习的效率和稳定性。

        # ------------------------------------------
        # 检查 GPU 是否可用
        self.device = scan_device(args.cpu)
        self.model.to(self.device)    # 将模型和优化器移动到 GPU

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)    # 定义了用于训练神经网络的优化器。这里使用的是Adam优化器，一个流行的梯度下降变种，lr=0.001设置了学习率为0.001。
        self.criterion = nn.MSELoss()    # 这定义了用于训练过程中的损失函数。这里使用的是均方误差损失（MSE Loss），它是评估神经网络预测值与实际值差异的常用方法。

        self.cur_episode = 0                # 回合数
        self.cur_epsilon = args.epsilon     # 探索率
        self.min_epsilon = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.render = args.render
        self.gamma = args.gamma
        self.batch_size = args.batch_size
    

    # 每轮训练后对探索率进行衰减
    # ε-贪婪策略（epsilon-greedy strategy）的强化学习技巧中的关键部分，用于平衡探索（exploration）和利用（exploitation）
    #   在强化学习中，智能体需要决定是利用当前已知的最佳策略（exploitation）来最大化短期奖励，还是探索新的动作（exploration）以获得更多信息，可能会带来更大的长期利益。
    #   ε-贪婪策略通过一个参数ε（epsilon）来控制这种平衡。ε的值是一个0到1之间的数字，表示选择随机探索的概率。
    def update_epsilon(self) :
        self.cur_epsilon = max(self.min_epsilon, self.epsilon_decay * self.cur_epsilon) # 衰减探索率
        return self.cur_epsilon
    

    def load_last_checkpoint(self) :
        if self.args.zero :
            return
        
        last_checkpoint = self.cp_mgr.load_last_checkpoint()
        if last_checkpoint :
            self.cur_episode = last_checkpoint.episode + 1
            self.epsilon = last_checkpoint.epsilon
            self.model.load_state_dict(last_checkpoint.model_state_dict)
            self.optimizer.load_state_dict(last_checkpoint.optimizer_state_dict)
        return
    


    def save_checkpoint(self, episode) :
        return self.cp_mgr.save_checkpoint(
            self.model, 
            self.optimizer, 
            self.cur_epsilon, 
            episode
        )
