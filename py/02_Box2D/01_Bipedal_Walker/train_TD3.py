#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/31 13:01
# -----------------------------------------------
# 经典控制： bipedal_walker （双足步行者）
#   bipedal_walker 是一个倒立摆摆动问题。
#   该系统由一个摆锤组成，摆锤的一端连接到固定点，另一端自由。
#   摆锤从一个随机位置开始，目标是在自由端施加扭矩，使其摆动到重心位于固定点的正上方的垂直位置，然后坚持得越久越好。
# 
# 相关文档：
#   https://gymnasium.farama.org/environments/box2d/bipedal_walker/
# -----------------------------------------------

# 添加公共库文件的相对位置
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../libs/')
# --------------------

import argparse
import torch
import torch.cuda
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import time
import random
import numpy as np
from bean.train_args import TrainArgs
from bean.transition import Transition
from utils.ctrl_env import CtrlInitEnv
from utils.terminate import TerminateDetector
from utils.adjust import *
from tools.utils import *
from conf.settings import *
from color_log.clog import log


def arguments() :
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='Gym - bipedal_walker 训练脚本',
        description='在默认环境下、使用深度 Q 网络（DQN）训练智能体操作 bipedal_walker', 
        epilog='\r\n'.join([
            '运行环境: python3', 
            '运行示例: python py/01_box2d/01_Bipedal_Walker/train_td3.py'
        ])
    )
    parser.add_argument('-u', '--human', dest='human', action='store_true', default=False, help='渲染模式: 人类模式，帧率较低且无法更改窗体显示内容')
    parser.add_argument('-a', '--rgb_array', dest='rgb_array', action='store_true', default=False, help='渲染模式: RGB 数组，需要用 OpenCV 等库辅助渲染，可以在每一帧添加定制内容，帧率较高')
    parser.add_argument('-s', '--save_gif', dest='save_gif', action='store_true', default=False, help='保存每个回合渲染的 UI 到 GIF（仅 rgb_array 模式有效）')
    parser.add_argument('-c', '--cpu', dest='cpu', action='store_true', default=False, help='强制使用 CPU: 默认情况下，自动优先使用 GPU 训练（除非没有 GPU）')
    parser.add_argument('-z', '--zero', dest='zero', action='store_true', default=False, help='强制从零开始重新训练（不加载上次训练的 checkpoint）')
    parser.add_argument('-e', '--epoches', dest='epoches', type=int, default=10000, help='训练次数: 即训练过程中智能体将经历的总回合数。每个回合是一个从初始状态到终止状态的完整序列')
    parser.add_argument('-g', '--gamma', dest='gamma', type=float, default=0.95, help='折扣因子: 用于折算未来奖励的在当前回合中的价值。一个接近 0 的值表示智能体更重视即时奖励，而接近 1 的值表示智能体更重视长期奖励。')
    parser.add_argument('-l', '--lr', dest='lr', type=float, default=0.001, help='学习率: 用于训练神经网络的优化器。对于 Adam 优化器使用 0.001 是最优的经验值。较高的学习率可能导致快速学习，但也可能导致过度调整，从而错过最佳值。较低的学习率意味着更慢的学习，但可以提高找到最优解的几率。')
    parser.add_argument('-p', '--epsilon', dest='epsilon', type=float, default=1.0, help='探索率: 用于 epsilon-greedy 策略，它决定了智能体探索新动作的频率。值越高，智能体越倾向于尝试新的、不确定的动作而不是已知的最佳动作。这个值通常在训练初期较高，随着学习的进行逐渐降低')
    parser.add_argument('-d', '--epsilon_decay', dest='epsilon_decay', type=float, default=0.995, help='衰减率: 探索率随时间逐渐减小的速率。每经过一个回合，epsilon 将乘以这个衰减率，从而随着时间的推移减少随机探索的频率')
    parser.add_argument('-m', '--min_epsilon', dest='min_epsilon', type=float, default=0.1, help='最小探索率: 即使经过多次衰减，探索率也不会低于这个值，确保了即使在后期也有一定程度的探索')
    parser.add_argument('-n', '--noise', dest='noise', type=float, default=0.1, help='TD3 算法的噪声强度，经验值。例如若动作空间的范围是 [-1, 1]，则噪声为 0.1 意味着在动作值的 10% 范围内波动')
    parser.add_argument('-o', '--noise_limit', dest='noise_limit', type=float, default=0.4, help='TD3 算法的噪声波动幅度，相对于动作空间范围设定。默认 0.4，即噪声如何波动都不会超出 [-0.4, 0.4] 的幅度')
    parser.add_argument('-w', '--tau', dest='tau', type=float, default=0.005, help='目标网络的更新率，决定了主网络对目标网络的影响程度，通常设置为一个很小的值')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32, help='从经验回放存储中一次抽取并用于训练网络的经验的样本数。默认值为 32，即每次训练时会使用 32 个经验样本')
    parser.add_argument('-t', '--tensor_logs', dest='tensor_logs', type=str, default=get_tensor_path(COURSE_NAME), help='TensorBoardX 日志目录')
    return parser.parse_args()


def main(args) :
    # 初始化训练环境和参数
    targs = TrainArgs(args)

    # 实现 “训练算法” 以进行训练
    # 针对 bipedal_walker 问题， TD3 算法会更适合：
    #   DDPG（Deep Deterministic Policy Gradient）：结合了策略梯度和 Q 学习的算法，特别适用于连续动作空间。
    #   TD3（Twin Delayed DDPG）：是 DDPG 的改进版本，通过使用两个 Q 网络和延迟策略更新来减少过高估计和提高稳定性。
    train_td3(targs)
    targs.close_env()


def train_td3(targs: TrainArgs) :
    '''
    使用 TD3 算法进行训练。
    :params: targs 用于训练的环境和模型关键参数
    :return: None
    '''
    writer = SummaryWriter(logdir=targs.tensor_logs) # 训练过程记录器，可用 TensorBoard 查看
    targs.load_last_checkpoint()                    # 加载最后一次训练的状态和参数

    log.info("++++++++++++++++++++++++++++++++++++++++")
    log.info("开始训练 ...")
    for epoch in range(targs.last_epoch, targs.epoches) :
        log.info(f"第 {epoch}/{targs.epoches} 回合训练开始 ...")
        train(writer, targs, epoch)

        targs.update_target_model(epoch)        # 更新目标模型
        epsilon = targs.update_epsilon()        # 衰减探索率
        targs.save_checkpoint(epoch, epsilon)   # 保存当次训练的状态和参数（用于断点训练）
        time.sleep(0.01)

    writer.close()
    log.warn("已完成全部训练")

    targs.save_checkpoint(targs.epoches, -1, True)
    log.info("----------------------------------------")



# FIXME 并发训练
def train(writer : SummaryWriter, targs : TrainArgs, epoch) :
    '''
    使用 TD3 算法进行训练。
    :params: writer 训练过程记录器，可用 TensorBoard 查看
    :params: targs 用于训练的环境和模型关键参数
    :return: None
    '''
    targs.reset_render_cache()

    cie = CtrlInitEnv(targs.env, targs.epoches) # 通过不断重试，获得各个阶段理想的初始状态，
    raw_obs = cie.reset(epoch)                  # 以使用“中间难度起点奖励(SoID)”策略帮智能体建立行动策略框架
    obs = to_tensor(raw_obs[0], targs, False)   # 把观测空间状态数组送入神经网络所在的设备
    
    total_reward = 0                # 累计智能体从环境中获得的总奖励。在每个训练回合结束时，total_reward 将反映智能体在该回合中的总体表现。奖励越高，意味着智能体的性能越好。
    total_loss = 0                  # 累计损失率。反映了预测 Q 值和目标 Q 值之间的差异
    step_counter = 0                # 训练步数计数器
    bgn_time = current_seconds()    # 训练时长计数器
    td = TerminateDetector()

    # 开始训练智能体
    while True:
        # 选择下一步动作
        action = select_next_action(targs, obs)

        # 执行下一步动作
        next_obs, reward, done = exec_next_action(targs, action, epoch, step_counter)

        # 调整奖励
        reward, terminated = adjust(next_obs, action, reward, td, step_counter)
        done = terminated or done

        # 向【经验回放存储】添加当前 step 执行前后状态、奖励情况等
        targs.memory.append((obs, action, reward, next_obs, done))

        obs = next_obs          # 更新当前状态
        total_reward += reward  # 累计奖励（每一步的奖励是 env 决定的，由于 env 使用默认环境，所以这里无法调整每一步的奖励）
        step_counter += 1       # 累计步数

        # 渲染训练时的 GUI （必须在 reset 方法后执行）
        labels = [
            f"epoch: {epoch}", 
            f"step: {step_counter}", 
            f"action: {action}", 
            f"total_reward: {total_reward}", 
            f"total_loss: {total_loss}",
            f"epsilon: {targs.cur_epsilon}", 
            f"coordinates-x: {obs[0][0]}",      # 自由端的 x 坐标
            f"coordinates-y: {obs[0][1]}",      # 自由端的 y 坐标
            f"angular_velocity: {obs[0][2]}",   # 角速度
        ]
        targs.render(labels)
        if done:
            break

        total_loss += td3(targs, step_counter)  # TD3 学习（核心算法，从【经验回放存储】中收集经验）
    # while end

    # 保存智能体这个回合渲染的动作 UI
    targs.save_render_ui(epoch)
    
    finish_time = current_seconds() - bgn_time
    log.info(f"第 {epoch}/{targs.epoches} 回合 完成，累计步数={step_counter} 步, 耗时={finish_time}s, 奖励={total_reward}, 探索率={targs.cur_epsilon}")

    # 记录每个回合结束时的训练参数到 TensorBoard
    # 分组名/指标 ， 分组名可以不要
    writer.add_scalar('通常/每回合探索率', targs.cur_epsilon, epoch)
    writer.add_scalar('通常/每回合步数', step_counter, epoch)
    writer.add_scalar('通常/每回合完成时间', finish_time, epoch)
    writer.add_scalar('通常/每回合总奖励', total_reward, epoch)
    writer.add_scalar('通常/每回合平均损失', total_loss / step_counter, epoch)
    writer.add_scalar('特殊/每回合学习率 (actor)', targs.actor_optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('特殊/每回合学习率 (critic)', targs.critic_optimizer.param_groups[0]['lr'], epoch)
    return



def select_next_action(targs: TrainArgs, obs) :
    '''
    选择下一步要执行的动作。
        在 TD3 中，动作的选择是通过 Actor 网络直接进行的，
        并且为了增加探索性，通常会向选定的动作添加一些噪声。
    :params: targs 用于训练的环境和模型关键参数
    :params: obs 当前观测空间的状态
    :return: 下一步要执行的动作
    '''

    # 在动作空间随机选择一个动作（受当前探索率影响）
    if not np.random.rand() <= targs.cur_epsilon:
        with torch.no_grad() :  # 暂时禁用梯度计算
            # 这是 PyTorch 的语法糖 actor_model(obs) == actor_model.forward(obs)
            action = targs.actor_model(obs).detach()    # array len 1
    else:
        action = targs.env.action_space.sample()        # array len 1

    # 在 TD3 中，为了探索，通常给动作添加一定的噪声
    noise = np.random.normal(0, targs.noise, size=targs.env.action_space.shape[0])  # array len 1
    action = add_noise(targs, action, noise)    # tensor len 1
    return to_nparray(action)     # array len 1



def exec_next_action(targs: TrainArgs, action, epoch=-1, step_counter=-1) :
    '''
    执行下一步动作
    :params: targs 用于训练的环境和模型关键参数
    :params: action 下一步动作
    :return: 执行动作后观测空间返回的状态
    '''
    # 旧版本的 env.step(action) 返回值只有 4 个参数，没有 truncated
    # 但是 truncated 和 info 暂时没用，详见 https://stackoverflow.com/questions/73195438/openai-gyms-env-step-what-are-the-values
    next_raw_obs, reward, terminated, truncated, info  = targs.env.step(action)
    # log.debug(f"[第 {epoch} 回合] 步数={step_counter}")
    # log.debug("执行结果：")
    # log.debug(f"  状态空间变化：{next_raw_obs}")  # 执行动作后的新状态或观察。这是智能体在下一个时间步将观察到的环境状态。
    # log.debug(f"  累计获得奖励：{reward}")        # 执行动作后获得的奖励。这是一个数值，指示执行该动作的效果好坏，是强化学习中的关键信号，作为当次动作的反馈。
    # log.debug(f"  回合是否结束: {terminated}")    # 可能成功也可能失败，例如在一些游戏中，达到目标或失败会结束回合。
    # log.debug(f"  回合是否中止: {truncated}")     # 回合因为某些约束调节提前中止，如步数限制等。
    # log.debug(f"  其他额外信息: {info}")          # 通常用 hash 表附带自定义的额外信息（如诊断信息、调试信息），暂时不需要用到的额外信息。
    
    next_obs = to_tensor(next_raw_obs, targs, False)      # 把观测空间状态数组送入神经网络所在的设备
    done = terminated or truncated
    return (next_obs, reward, done)



def td3(targs: TrainArgs, step_counter) :
    '''
    进行 TD3 学习（基于 Q 值的强化学习方法）：
        这个过程是 TD3 学习算法的核心，它利用从环境中收集的经验来不断调整和优化网络，使得预测的 Q 值尽可能接近实际的 Q 值。
        通过迭代这个过程，使得神经网络逐渐学习到一个策略，该策略可以最大化累积奖励。
    :params: targs 用于训练的环境和模型关键参数
    :params: action 下一步动作
    :params: step_counter 步数计数器
    :return: 执行动作后观测空间返回的状态
    '''
    total_loss = 0

    # 确保只有当【经验回放存储】中的样本数量超过批处理大小时，才进行学习过程
    # 这是为了确保有足够的样本来进行有效的批量学习
    if len(targs.memory) <= targs.batch_size :
        return total_loss
    
    # ===============================
    # 准备批量数据
    # ===============================
    # 从【经验回放存储】中随机抽取 batch_size 个样本数
    # 这种随机抽样是为了减少样本间的相关性，增强学习的稳定性和效率
    transitions = random.sample(targs.memory, targs.batch_size)

    # 解压 transitions 到单独的批次
    batch = Transition(*zip(*transitions))

    # 将每个样本的组成部分 (obs, action, reward, next_obs, done) ，拆分转换为独立的批次
    obs_batch = down_dim(torch.stack(batch.obs).to(targs.device))
    act_batch = torch.tensor(np.array(batch.action), dtype=torch.float, device=targs.device)
    reward_batch = up_dim(torch.tensor(batch.reward, dtype=torch.float, device=targs.device))
    next_obs_batch = down_dim(torch.stack(batch.next_obs).to(targs.device))
    done_batch = up_dim(torch.tensor(batch.done, dtype=torch.float, device=targs.device))

    # print(f"obs_batch: {obs_batch}")              # tensor 32x2
    # print(f"act_batch: {act_batch}")              # tensor 32x1
    # print(f"reward_batch: {reward_batch}")        # tensor 32x1
    # print(f"next_obs_batch: {next_obs_batch}")    # tensor 32x2
    # print(f"done_batch: {done_batch}")            # tensor 32x1
    

    # ===============================
    # 更新 Critic 网络
    # ===============================

    # 计算下一个状态的最大预测 Q 值
    with torch.no_grad():
        next_act_batch = targs.target_actor_model(next_obs_batch)  # tensor 32x1
        noise_batch = (torch.randn_like(next_act_batch) * targs.noise).clip(-targs.noise_limit, targs.noise_limit) 
        next_act_batch = add_noise(targs, next_act_batch, noise_batch)   # tensor 32x1

        # next_obs_batch 必须形状 tensor 32x2
        # next_act_batch 必须形状 tensor 32x1
        target_Q1, target_Q2 = targs.target_critic_model(next_obs_batch, next_act_batch)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q_values = reward_batch + (1 - done_batch) * targs.gamma * target_Q


    # 获得当前状态下的 Q 值（对当前状态的观测进行前向传播的结果，用于计算损失）
    # obs_batch 必须形状 tensor 32x2
    # act_batch 必须形状 tensor 32x1
    current_Q1, current_Q2 = targs.critic_model(obs_batch, act_batch)
    critic_loss = F.mse_loss(current_Q1, target_Q_values) + F.mse_loss(current_Q2, target_Q_values)
    total_loss += critic_loss.item()    # 更新累积损失
    optimize_params(targs.critic_model, targs.critic_optimizer, critic_loss)    # 参数优化


    # ===============================
    # 延迟更新 Actor 网络
    # ===============================
    # 在 TD3 中，Actor 网络的更新频率较低（例如，每 2 次 Critic 更新后更新一次 Actor），以减少策略的方差。
    if step_counter % targs.update_action_every == 0 :
        actor_loss = -targs.critic_model.Q1(obs_batch, targs.actor_model(obs_batch)).mean()
        total_loss += actor_loss.item()     # 更新累积损失
        optimize_params(targs.actor_model, targs.actor_optimizer, actor_loss)   # 参数优化

    return total_loss


def add_noise(targs: TrainArgs, action, noise) :
    '''
    为 action 添加噪音：TD3 通过在选取的动作上添加噪声来平滑目标策略
    :params: targs 用于训练的环境和模型关键参数
    :params: action 动作张量        array or tensor
    :params: noise 噪音张量         array or tensor
    :return: 添加噪音的动作张量      tensor （维度和形状 与 action和noise 一致）
    '''
    min_action = torch.tensor(targs.env.action_space.low, device=targs.device, dtype=torch.float)
    max_action = torch.tensor(targs.env.action_space.high, device=targs.device, dtype=torch.float)
    action = (action + noise).clip(min_action, max_action) # 确保 action + noise 依然在动作空间的取值范围内
    return action


def optimize_params(model, optimizer, loss, max_grad_norm=1.0) :
    '''
    优化模型
    :params: model 模型
    :params: optimizer 优化器
    :params: loss 损失
    :params max_grad_norm: 梯度的最大范数
    :return: 
    '''
    optimizer.zero_grad() # 清除之前的梯度。PyTorch 会默认累积梯度，如果不清除梯度，新计算的梯度会被加到已存在的梯度上，在 DQN 中这会使得训练过程变得不稳定，甚至可能导致模型完全无法学习。
    loss.backward()       # 反向传播，计算梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 梯度裁剪
    optimizer.step()      # 更新参数（梯度下降，指使用计算出的梯度来更新模型参数的过程）



if __name__ == '__main__' :
    main(arguments())

