#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from bean.dqn import DQN
from collections import deque
from tools.utils import scan_device
from bean.checkpoint import CheckpointManager
from conf.settings import *


class TrainArgs :

    def __init__(self, args, env, eval=False, 
                 checkpoints_dir=CHECKPOINTS_DIR, 
                 save_interval=SAVE_CHECKPOINT_INTERVAL) -> None:
        '''
        初始化深度 Q 网络（DQN）算法的环境和模型关键参数。
        :params: args 从命令行传入的训练控制参数
        :params: env 当前交互的环境变量，如 Acrobot
        :params: eval 评估模式，仅用于验证模型
        :params: checkpoints_dir 存储检查点的目录
        :params: save_interval 存储检查点的回合数间隔
        :return: TrainArgs
        '''
        self.args = args
        self.env = env
        self.render = args.render                           # 渲染 GUI 开关
        
        self.obs_size = env.observation_space.shape[0]      # 状态空间维度
        self.action_size = env.action_space.n               # 动作空间数量

        self.model = DQN(self.obs_size, self.action_size)   # DQN 简单的三层网络模型（主模型）
        self.device = scan_device(args.cpu)                 # 检查 GPU 是否可用
        self.model.to(self.device)                          # 将模型和优化器移动到 GPU （或 CPU）

        if eval :
            self.model.eval()   # 评估模式

        else :
            self.cp_mgr = CheckpointManager(            # checkpoint 管理器
                checkpoints_dir, 
                save_interval
            )  

            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # 用于训练神经网络的优化器。这里使用的是Adam优化器，一个流行的梯度下降变种，lr=0.001设置了学习率为0.001。
            self.criterion = nn.MSELoss()                                   # 用于训练过程中的损失函数。这里使用的是均方误差损失（MSE Loss），它是评估神经网络预测值与实际值差异的常用方法。

            self.memory = deque(maxlen=2000)            # 经验回放存储。本质是一个双端队列（deque），当存储超过2000个元素时，最旧的元素将被移除。经验回放是DQN中的一项关键技术，有助于打破经验间的相关性并提高学习的效率和稳定性。
            self.batch_size = args.batch_size           # 从【经验回放存储】中一次抽取并用于训练网络的【经验样本数】

            self.epoches = args.epoches                 # 总训练回合数
            self.last_epoch = 0                         # 最后一次记录的训练回合数
            self.zero = self.args.zero                  # 强制从第 0 回合开始训练
            self.cur_epsilon = args.epsilon             # 当前探索率
            self.epsilon_decay = args.epsilon_decay     # 探索率的衰减率
            self.min_epsilon = args.min_epsilon         # 最小探索率
            self.gamma = args.gamma                     # 折扣因子
            self.info = {}                              # 其他额外参数

            # 在 DQN 中，通常会使用两个模型：
            #   一个是用于进行实际决策的主模型（self.model）： 用于生成当前的 Q 值
            #   另一个是目标模型（target_model）：用于计算期望的 Q 值，以提供更稳定的学习目标
            self.target_model = DQN(self.obs_size, self.action_size)    # 目标模型
            self.target_model.to(self.device)                           # 将模型移动到 GPU （或 CPU）
            self.update_target_every = 5                                # 定义更新目标模型的频率


    def update_target_model(self, epoch):
        '''
        使用 主模型 更新 目标模型 网络的参数。
            在训练循环中，需要定期更新目标模型的参数，这通常在固定的回合数之后发生。
            而且两个模型的参数不会同时更新，学习过程会更加稳定。
        :params: epoch 已训练回合数
        :return: None
        '''
        if epoch % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        
    def load_last_checkpoint(self) :
        '''
        加载最后一次记录的训练检查点
        :return: None
        '''
        if self.zero :
            return  # 强制从零开始训练，不加载检查点
        
        last_cp = self.cp_mgr.load_last_checkpoint()
        if last_cp :
            self.last_epoch = last_cp.epoch + 1
            self.cur_epsilon = last_cp.epsilon
            self.info = last_cp.info
            self.model.load_state_dict(last_cp.model_state_dict)
            self.optimizer.load_state_dict(last_cp.optimizer_state_dict)
        return
    


    def save_checkpoint(self, epoch, epsilon, info={}) :
        '''
        保存训练检查点。
        但是若未满足训练回合数，不会进行保存。
        :params: epoch 已训练回合数
        :params: epsilon 当前探索率
        :params: info 其他附加参数
        :return: 是否保存了检查点
        '''
        return self.cp_mgr.save_checkpoint(
            self.model, 
            self.optimizer, 
            epoch, 
            epsilon, 
            info
        )



    # 每轮训练后对探索率进行衰减。
    #   ε-贪婪策略（epsilon-greedy strategy）的强化学习技巧中的关键部分，用于平衡探索（exploration）和利用（exploitation）
    #   在强化学习中，智能体需要决定是利用当前已知的最佳策略（exploitation）来最大化短期奖励，还是探索新的动作（exploration）以获得更多信息，可能会带来更大的长期利益。
    #   ε-贪婪策略通过一个参数ε（epsilon）来控制这种平衡。ε的值是一个0到1之间的数字，表示选择随机探索的概率。
    def update_epsilon(self) :
        '''
        使用衰减率对探索率进行衰减
        :return: 衰减一次后的探索率
        '''
        epsilon = max(
            self.min_epsilon, 
            self.cur_epsilon * self.epsilon_decay
        )
        self.cur_epsilon = epsilon
        return epsilon
