#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/17 20:33
# -----------------------------------------------

import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from bean.actor import Actor
from bean.critic import Critic
from collections import deque
from tools.utils import scan_device
from bean.tagger import Tagger
from bean.checkpoint import CheckpointManager
from conf.settings import *


class TrainArgs :

    def __init__(self, args, eval=False) -> None:
        '''
        初始化深度 Q 网络（DQN）算法的环境和模型关键参数。
        :params: args 从命令行传入的训练控制参数
        :params: env 当前交互的环境变量
        :params: eval 评估模式，仅用于验证模型
        :return: TrainArgs
        '''
        self.args = args
        self.env = self.create_env(ENV_NAME)
        
        self.obs_size = self.env.observation_space.shape[0]     # 状态空间维度
        self.act_size = self.env.action_space.shape[0]          # 动作空间维度
        self.max_action = float(self.env.action_space.high[0])  # 最大动作值
        self.device = scan_device(args.cpu)                     # 检查使用 GPU 还是 CPU

        # TD3 的 Actor-Critic 网络模型（主模型）
        self.models = self.create_models(self.obs_size, self.act_size, self.max_action, self.device)
        self.actor_model = self.models[0]
        self.critic_model = self.models[1]

        if eval :
            self.tagger = Tagger(COURSE_NAME, MODEL_NAME, ENV_NAME, True)
            self.eval(self.models)  # 评估模式

        else :
            self.tagger = Tagger(COURSE_NAME, MODEL_NAME, ENV_NAME, False)
            self.model_names = [ ACTOR_MODEL_NAME, CRITIC_MODEL_NAME ]
            self.mgrs = self.create_checkpoints(COURSE_NAME, self.model_names)  # checkpoint 管理器

            self.optimizers = self.create_optimizers(self.models, args.lr)  # 用于训练神经网络的优化器。这里使用的是 Adam 优化器，一个流行的梯度下降变种，lr=0.001设置了学习率为0.001
            self.actor_optimizer = self.optimizers[0]
            self.critic_optimizer = self.optimizers[1]
            self.criterion = nn.MSELoss()               # 用于训练过程中的损失函数。这里使用的是均方误差损失（MSE Loss），它是评估神经网络预测值与实际值差异的常用方法。

            self.memory = deque(maxlen=2000)            # 经验回放存储。本质是一个双端队列（deque），当存储超过2000个元素时，最旧的元素将被移除。经验回放是DQN中的一项关键技术，有助于打破经验间的相关性并提高学习的效率和稳定性。
            self.batch_size = args.batch_size           # 从【经验回放存储】中一次抽取并用于训练网络的【经验样本数】
            self.update_action_every = 2                # 定义更新 Action 模型的频率（一般 2-5，小值加速学习、大值则更稳定）

            self.epoches = args.epoches                 # 总训练回合数
            self.last_epoch = 0                         # 最后一次记录的训练回合数
            self.zero = self.args.zero                  # 强制从第 0 回合开始训练
            self.cur_epsilon = args.epsilon             # 当前探索率
            self.epsilon_decay = args.epsilon_decay     # 探索率的衰减率
            self.min_epsilon = args.min_epsilon         # 最小探索率
            self.noise = args.noise                     # 噪声强度
            self.noise_limit = args.noise_limit         # 噪声波动幅度
            self.tau = args.tau                         # 目标网络的更新率
            self.gamma = args.gamma                     # 折扣因子
            self.info = {}                              # 其他额外参数
            self.tensor_logs = args.tensor_logs         # TensorBoard 日志目录

            # TD3 与 DQN 类似，也是使用两个模型：
            #   一个是用于进行实际决策的主模型（self.model）： 用于生成当前的 Q 值
            #   另一个是目标模型（target_model）：用于计算期望的 Q 值，以提供更稳定的学习目标
            self.target_models = self.create_models(
                self.obs_size, self.act_size, self.max_action, self.device
            )
            self.target_actor_model = self.target_models[0]
            self.target_critic_model = self.target_models[1]
            self.update_target_every = 5                # 定义更新目标模型的频率
            


    def create_env(self, env_name) :
        '''
        创建和配置环境
        https://gymnasium.farama.org/api/env/
        https://panda-gym.readthedocs.io/en/latest/usage/advanced_rendering.html
        :return: 预设环境
        '''
        # 注意训练时尽量不要渲染 GUI，会极其影响训练效率
        if self.args.human :
            env = gym.make(env_name, render_mode="human")

        elif self.args.rgb_array :
            env = gym.make(env_name, render_mode="rgb_array")

        else :
            env = gym.make(env_name)
        return env


    def reset_env(self) :
        '''
        重置预设环境
        :return: None
        '''
        return self.env.reset()


    def close_env(self) :
        '''
        关闭预设环境
        :return: None
        '''
        self.env.close()


    def render(self, labels=[]) :
        '''
        渲染 UI
        :params: labels 渲染 UI 时附加到左上角的信息（仅 rgb_array 模式下有效）
        :return: frame 渲染的当前帧（若纯后台执行返回 None）
        '''
        frame = None
        if self.args.human :
            frame = self.env.render()

        elif self.args.rgb_array :
            frame = self.env.render()
            if self.tagger.show(frame, labels) :
                os._exit(0)
        return frame
    

    def reset_render_cache(self) :
        '''
        重置渲染画面的内存
        :return: None
        '''
        if self.args.save_gif :
            self.tagger.reset()
    

    def save_render_ui(self, epoch) :
        '''
        保存智能体第 epoch 回合渲染的动作 UI 到 GIF
        :params: epoch 回合数
        :return: None
        '''
        if self.args.save_gif :
            self.tagger.save_ui(epoch)
    

    def create_models(self, obs_size, act_size, max_action, device) :
        '''
        构建 TD3 模型
        :params: obs_size 状态空间维度
        :params: act_size 动作空间维度
        :params: max_action 最大动作值
        :params: device 设备（GPU/CPU）
        :return: None
        '''
        actor_model = Actor(obs_size, act_size, max_action) # 策略模型
        critic_model = Critic(obs_size, act_size)           # Q 值模型

        # 将模型和优化器移动到 GPU （或 CPU）
        actor_model.to(device)               
        critic_model.to(device) 
        return [actor_model, critic_model]
    

    def create_optimizers(self, models, lr) :
        '''
        构建优化器
        :params: models 模型列表
        :params: lr 学习率
        :return: 优化器列表
        '''
        optimizers = []
        for model in models :
            optimizer = optim.Adam(model.parameters(), lr=lr)
            optimizers.append(optimizer)
        return optimizers
    

    def eval(self, models) :
        '''
        评估模式
        :params: models 模型列表
        :return: None
        '''
        for model in models :
            model.eval()


    def load_models(self, model_paths=[]) :
        '''
        加载已训练好的模型
        :params: model_paths 模型路径列表
        :return: None
        '''
        for model_path in model_paths :
            if ACTOR_MODEL_NAME in model_path :
                self.actor_model.load_state_dict(         
                    torch.load(model_path)
                )
            elif CRITIC_MODEL_NAME in model_path :
                self.critic_model.load_state_dict(         
                    torch.load(model_path)
                )
    

    def update_target_model(self, epoch):
        '''
        使用 主模型 更新 目标模型 网络的参数。
            在训练循环中，需要定期更新目标模型的参数，这通常在固定的回合数之后发生。
            而且两个模型的参数不会同时更新，学习过程会更加稳定。

        在 TD3 中，目标网络的更新使用了软更新策略，即目标网络的参数是主网络参数和旧目标网络参数的加权平均。
        软更新策略提供了一种渐进式的更新方式，它可以减少目标网络参数在更新时的突变，从而增强整个学习过程的稳定性。
        
        :params: epoch 已训练回合数
        :return: None
        '''
        if epoch % self.update_target_every == 0:
            for idx, target_model in enumerate(self.target_models) :
                for target_param, param in zip(target_model.parameters(), self.models[idx].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



    def create_checkpoints(self, course_name, model_names) :
        '''
        创建检查点管理器
        :params: course_name 课程名称
        :params: model_names 模型列表
        :return: 管理器列表
        '''
        mgrs = []
        for model_name in model_names :
            mgr = CheckpointManager(course_name, model_name)
            mgrs.append(mgr)
        return mgrs

        
    def load_last_checkpoint(self) :
        '''
        加载最后一次记录的训练检查点
        :return: None
        '''
        if self.zero :
            return  # 强制从零开始训练，不加载检查点
        
        for idx, mgr in enumerate(self.mgrs) :
            last_cp = mgr.load_last_checkpoint(self.model_names[idx])
            if last_cp :
                self.last_epoch = last_cp.epoch + 1
                self.cur_epsilon = last_cp.epsilon
                self.info = last_cp.info
                self.models[idx].load_state_dict(last_cp.model_state_dict)
                self.optimizers[idx].load_state_dict(last_cp.optimizer_state_dict)
        return


    def save_checkpoint(self, epoch, epsilon, info={}, force=False) :
        '''
        保存训练检查点。
        但是若未满足训练回合数，不会进行保存。
        :params: epoch 已训练回合数
        :params: epsilon 当前探索率
        :params: info 其他附加参数
        :params: force 强制保存
        :return: 是否保存了检查点
        '''
        if force and (epsilon < 0) :
            epsilon = self.cur_epsilon

        is_ok = True
        for idx, mgr in enumerate(self.mgrs) :
            is_ok &= mgr.save_checkpoint(
                self.models[idx], 
                self.optimizers[idx], 
                epoch, 
                epsilon, 
                info
            )
        return is_ok


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
