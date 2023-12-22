#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------
# 经典控制： Cart Pole （倒立摆）
#   Cart Pole 是一个倒立摆问题，目标是控制一个小车来平衡一个竖直安装在其上的杆子。
# 
# 相关文档：
#   https://gymnasium.farama.org/environments/classic_control/cart_pole/
#   http://incompleteideas.net/book/11/node4.html
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
import random
import numpy as np
import gymnasium as gym
from bean.train_args import TrainArgs
from bean.transition import Transition
from tools.utils import *
from conf.settings import *
from color_log.clog import log


def arguments() :
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='Gym - CartPole 训练脚本',
        description='在默认环境下、使用深度 Q 网络（DQN）训练智能体操作 CartPole', 
        epilog='\r\n'.join([
            '运行环境: python3', 
            '运行示例: python py/01_Classic_Control/02_Cart_Pole/train_DQN.py'
        ])
    )
    parser.add_argument('-r', '--render', dest='render', action='store_true', default=False, help='渲染模式: 可以通过 GUI 观察智能体实时交互情况，但是会极大拉低训练效率')
    parser.add_argument('-c', '--cpu', dest='cpu', action='store_true', default=False, help='强制使用 CPU: 默认情况下，自动优先使用 GPU 训练（除非没有 GPU）')
    parser.add_argument('-z', '--zero', dest='zero', action='store_true', default=False, help='强制从零开始重新训练（不加载上次训练的 checkpoint）')
    parser.add_argument('-e', '--epoches', dest='epoches', type=int, default=10000, help='训练次数: 即训练过程中智能体将经历的总回合数。每个回合是一个从初始状态到终止状态的完整序列')
    parser.add_argument('-g', '--gamma', dest='gamma', type=float, default=0.95, help='折扣因子: 用于折算未来奖励的在当前回合中的价值。一个接近 0 的值表示智能体更重视即时奖励，而接近 1 的值表示智能体更重视长期奖励。')
    parser.add_argument('-l', '--lr', dest='lr', type=float, default=0.001, help='学习率: 用于训练神经网络的优化器。对于 Adam 优化器使用 0.001 是最优的经验值。较高的学习率可能导致快速学习，但也可能导致过度调整，从而错过最佳值。较低的学习率意味着更慢的学习，但可以提高找到最优解的几率。')
    parser.add_argument('-s', '--epsilon', dest='epsilon', type=float, default=1.0, help='探索率: 用于 epsilon-greedy 策略，它决定了智能体探索新动作的频率。值越高，智能体越倾向于尝试新的、不确定的动作而不是已知的最佳动作。这个值通常在训练初期较高，随着学习的进行逐渐降低')
    parser.add_argument('-d', '--epsilon_decay', dest='epsilon_decay', type=float, default=0.995, help='衰减率: 探索率随时间逐渐减小的速率。每经过一个回合，epsilon 将乘以这个衰减率，从而随着时间的推移减少随机探索的频率')
    parser.add_argument('-m', '--min_epsilon', dest='min_epsilon', type=float, default=0.1, help='最小探索率: 即使经过多次衰减，探索率也不会低于这个值，确保了即使在后期也有一定程度的探索')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32, help='从经验回放存储中一次抽取并用于训练网络的经验的样本数。默认值为 32，即每次训练时会使用 32 个经验样本')
    parser.add_argument('-t', '--tensor_logs', dest='tensor_logs', type=str, default=get_tensor_path(MODEL_NAME), help='TensorBoardX 日志目录')
    return parser.parse_args()


def main(args) :
    # 创建和配置环境
    env = gym.make(ENV_NAME, 
        # 注意训练时尽量不要渲染 GUI，会极其影响训练效率
        render_mode=("human" if args.render else None)
    )

    # 实现 “训练算法” 以进行训练
    # 针对 CartPole 问题， DQN 算法会更适合：
    #   DQN（Deep Q-Network）是一种将深度学习与强化学习相结合的算法
    #   它主要用于解决具有连续、高维状态空间的问题，特别是那些传统的 Q-learning 算法难以处理的问题。
    #   在 DQN 中，传统 Q-learning 中的 Q 表（一个用于存储所有状态-动作对应价值的巨大表格）被一个深度神经网络所替代。
    #   这个神经网络被训练来预测给定状态和动作下的 Q 值
    train_dqn(args, env)


def train_dqn(args, env) :
    '''
    使用深度 Q 网络（DQN）算法进行训练。
    :params: args 从命令行传入的训练控制参数
    :params: env 当前交互的环境变量，如 CartPole
    :return: None
    '''
    writer = SummaryWriter(logdir=args.tensor_logs) # 训练过程记录器，可用 TensorBoard 查看
    targs = TrainArgs(args, env)                    # 初始化训练参数
    targs.load_last_checkpoint()                    # 加载最后一次训练的状态和参数

    log.info("++++++++++++++++++++++++++++++++++++++++")
    log.info("开始训练 ...")
    for epoch in range(targs.last_epoch, args.epoches) :
        log.info(f"第 {epoch}/{args.epoches} 回合训练开始 ...")
        train(writer, targs, epoch)

        targs.update_target_model(epoch)        # 更新目标模型
        epsilon = targs.update_epsilon()        # 衰减探索率
        targs.save_checkpoint(epoch, epsilon)   # 保存当次训练的状态和参数（用于断点训练）

    writer.close()
    env.close()
    log.warn("已完成全部训练")

    targs.save_checkpoint(args.epoches, -1, True)
    log.info("----------------------------------------")



# FIXME 并发训练
def train(writer : SummaryWriter, targs : TrainArgs, epoch) :
    '''
    使用深度 Q 网络（DQN）算法进行训练。
    :params: writer 训练过程记录器，可用 TensorBoard 查看
    :params: targs 用于训练的环境和模型关键参数
    :return: None
    '''
    raw_obs = targs.env.reset()         # 重置环境（在 CartPole 环境中，这个初始状态就是观测空间，它包含了关于 CartPole 状态的数组）
                                        # raw_obs 的第 0 个元素才是状态数组 (array([ 0.9996459 ,  0.02661069,  0.9958208 ,  0.09132832, -0.04581745, -0.06583451], dtype=float32), {})
    obs = to_tensor(raw_obs[0], targs)  # 把观测空间状态数组送入神经网络所在的设备
    
    # 渲染训练时的 GUI （必须在 reset 方法后执行）
    if targs.render :
        targs.env.render()
    
    total_reward = 0                # 累计智能体从环境中获得的总奖励。在每个训练回合结束时，total_reward 将反映智能体在该回合中的总体表现。奖励越高，意味着智能体的性能越好。
    total_loss = 0                  # 累计损失率。反映了预测 Q 值和目标 Q 值之间的差异
    step_counter = 0                # 训练步数计数器
    bgn_time = current_seconds()    # 训练时长计数器

    # 开始训练智能体
    while True:
        # 选择下一步动作
        action = select_next_action(targs, obs)

        # 执行下一步动作
        next_obs, reward, done = exec_next_action(targs, action, epoch, step_counter)
        
        # 向【经验回放存储】添加当前 step 执行前后状态、奖励情况等
        targs.memory.append((obs, action, reward, next_obs, done))

        obs = next_obs          # 更新当前状态
        total_reward += reward  # 累计奖励（每一步的奖励是 env 决定的，由于 env 使用默认环境，所以这里无法调整每一步的奖励）
        step_counter += 1       # 累计步数
        if done:
            break

        dqn(targs, total_loss)  # DQN 学习（核心算法，从【经验回放存储】中收集经验）
    # while end

    finish_time = current_seconds() - bgn_time
    log.info(f"第 {epoch}/{targs.epoches} 回合 完成，累计步数={step_counter} 步, 耗时={finish_time}s, 奖励={total_reward}, 探索率={targs.cur_epsilon}")

    # 记录每个回合结束时的训练参数到 TensorBoard
    # 分组名/指标 ， 分组名可以不要
    writer.add_scalar('通常/每回合探索率', targs.cur_epsilon, epoch)
    writer.add_scalar('通常/每回合步数', step_counter, epoch)
    writer.add_scalar('通常/每回合完成时间', finish_time, epoch)
    writer.add_scalar('通常/每回合总奖励', total_reward, epoch)
    writer.add_scalar('通常/每回合平均损失', total_loss / step_counter, epoch)
    writer.add_scalar('特殊/每回合学习率', targs.optimizer.param_groups[0]['lr'], epoch)
    writer.add_histogram('特殊/每回合 Q 值分布', targs.model(obs), epoch)     # 用于了解模型对每个状态-动作对的估计
    return



def select_next_action(targs: TrainArgs, obs) :
    '''
    选择下一步要执行的动作。
        DQN 一般使用 epsilon-greedy 策略：
        智能体在选择动作时，有一定概率随机探索环境，而其余时间则根据已学习的策略选择最佳动作
    :params: targs 用于训练的环境和模型关键参数
    :params: obs 当前观测空间的状态
    :return: 下一步要执行的动作
    '''

    # 在观测空间随机选择一个动作（受当前探索率影响）
    if np.random.rand() <= targs.cur_epsilon :
        action = targs.env.action_space.sample()  

    # 智能体根据当前经验、选择当前估计最优的动作
    else :
        action = torch.argmax(      # 2. 使用torch.argmax选择具有最高预期Q值的动作（当前状态下的最佳动作）
            targs.model(obs)        # 1. 将观测空间当前状态 obs （张量）传递给神经网络模型（model），神经网络模型输出每个可能动作的预期Q值
        ).item()                    # 3. 从张量中提取动作值
    return action



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
    
    next_obs = to_tensor(next_raw_obs, targs)      # 把观测空间状态数组送入神经网络所在的设备
    done = terminated or truncated                 # 在 CartPole 问题中，如果超过 500 步未达成目标就会中止游戏
    return (next_obs, reward, done)


def dqn(targs: TrainArgs, total_loss) :
    '''
    进行 DQN 学习（基于 Q 值的强化学习方法）：
        这个过程是 DQN 学习算法的核心，它利用从环境中收集的经验来不断调整和优化网络，使得预测的 Q 值尽可能接近实际的 Q 值。
        通过迭代这个过程，使得神经网络逐渐学习到一个策略，该策略可以最大化累积奖励。
    :params: targs 用于训练的环境和模型关键参数
    :params: action 下一步动作
    :return: 执行动作后观测空间返回的状态
    '''

    # 确保只有当【经验回放存储】中的样本数量超过批处理大小时，才进行学习过程
    # 这是为了确保有足够的样本来进行有效的批量学习
    if len(targs.memory) <= targs.batch_size :
        return
    
    # 从【经验回放存储】中随机抽取 batch_size 个样本数
    # 这种随机抽样是为了减少样本间的相关性，增强学习的稳定性和效率
    transitions = random.sample(targs.memory, targs.batch_size)

    # 解压 transitions 到单独的批次
    batch = Transition(*zip(*transitions))

    # 将每个样本的组成部分 (obs, action, reward, next_obs, done) ，拆分转换为独立的批次
    # 目的是后续计算时可以批量进行、加速运算 （单个计算也可以，但是太慢了）
    obs_batch = cat_batch_tensor(batch.obs, torch.float32)
    action_batch = cat_batch_tensor(batch.action, torch.long, up_dim=True)
    reward_batch = cat_batch_tensor(batch.reward, torch.float32, up_dim=True)
    next_obs_batch = cat_batch_tensor(batch.next_obs, torch.float32)
    done_batch = cat_batch_tensor(batch.done, torch.float32, up_dim=True)

    # 获得当前状态下的 Q 值（对当前状态的观测进行前向传播的结果，用于后续计算损失）
    current_Q_values = get_current_Q_values(targs.model, obs_batch, action_batch)

    # 计算下一个状态的最大预测 Q 值
    expected_q_values = calculate_expected_Q_values(
        targs.target_model, targs.gamma, 
        next_obs_batch, reward_batch, done_batch
    )

    # 计算 Huber 损失（亦称平滑 L1 损失，用于表示当前 Q 值和预期 Q 值之间的差异）
    # Huber 损失是均方误差和绝对误差的结合，对异常值不那么敏感
    loss = F.smooth_l1_loss(
        current_Q_values,               # 是一个二维张量，其形状是 [batch_size, 1]，因为它是通过 gather 操作从网络输出中选取的特定动作的 Q 值
        expected_q_values.unsqueeze(1)  # 在张量中增加一个维度，其中第二个维度（列）的大小为 1，即使其形状从 [batch_size] 转换成 [batch_size, 1]
    )

    # 优化模型
    optimize_params(targs.model, targs.optimizer, loss)

    # 累积损失更新
    total_loss += loss.item()


def cat_batch_tensor(batch_data, data_type, up_dim=False) :
    '''
    连接一批张量
    :params: batch_data 一批张量数据
    :params: data_type 张量元素的数据类型
    :params: up_dim 是否需要升维。当且仅当张量数据是 0 维标量时，才需要升维，否则 torch.cat 会报错
    :return: 连接后的张量
    '''
    batch_tensor = [torch.tensor([d], dtype=data_type) for d in batch_data] \
                if up_dim else \
            [d.clone().detach() for d in batch_data]
    return torch.cat(batch_tensor)


def get_current_Q_values(model, obs_batch, action_batch) :
    '''
    获取当前状态下的 Q 值（真实值）
    :params: model 主网络/主模型。在 DQN 算法中，通常使用两个网络：主网络（用于选择动作和更新），目标网络（用于计算目标 Q 值）。
    :params: obs_batch 观测空间的当前状态批量数据。
    :params: action_batch 动作空间的批量数据
    :return: 
    '''

    # 步骤 1: 将观测数据（状态）输入到模型中，以获取每个状态下所有动作的预测 Q 值。
    # obs_batch 是当前状态的批量数据
    predicted_Q_values = model(obs_batch)

    # 步骤 2: 对动作张量进行处理，以使其维度与预测的 Q 值张量匹配。
    # action_batch 包含了每个状态下选择的动作。
    # unsqueeze(1) 是将 action_batch 从 [batch_size] 转换为 [batch_size, 1]
    # 这是为了在接下来的操作中能够选择正确的 Q 值。
    actions_unsqueezed = action_batch.unsqueeze(1)

    # 步骤 3: 从预测的 Q 值中选择对应于实际采取的动作的 Q 值。
    # gather 函数在这里用于实现这一点。
    # 第一个参数 1 表示操作在第二维度（动作维度）上进行。
    # actions_unsqueezed 用作索引，指定从每一行（每个状态）中选择哪个动作的 Q 值。
    cur_Q_values = predicted_Q_values.gather(1, actions_unsqueezed)
    return cur_Q_values


def calculate_expected_Q_values(target_model, gamma, next_obs_batch, reward_batch, done_batch) :
    '''
    计算下一个状态的最大预测 Q 值
    :params: target_model 目标网络/目标模型。在 DQN 算法中，通常使用两个网络：主网络（用于选择动作和更新），目标网络（用于计算目标 Q 值）。目标网络的参数定期从主网络复制过来，但在更新间隔内保持不变。这有助于稳定学习过程。
    :params: gamma 折扣因子: 用于折算未来奖励的在当前回合中的价值。
    :params: next_obs_batch 执行动作后、观测空间的状态批量数据。
    :params: reward_batch 执行动作后、获得奖励的批量数据
    :params: done_batch 回合完成状态的批量数据
    :return: 
    '''

    # target_model(next_obs_batch): 将下一个状态的批量数据输入目标网络，得到每个状态下每个可能动作的预测 Q 值。
    # detach(): 用于切断这些 Q 值与计算图的联系。它会创建一个新的张量，它与原始数据共享内容，但不参与梯度计算。这在计算目标 Q 值时很常见，因为我们不希望在反向传播中更新目标网络。
    # max(1)：用于找出每个状态的所有动作 Q 值中的最大值。.max(1) 的 1 表示操作是在张量的第二个维度上进行的（即动作维度）
    # [0]：从 .max(1) 返回的结果中，只取最大值。因为 .max(1) 返回一个元组，其中第一个元素（索引为 0 的元素）是每一行的最大值。
    next_Q_values = target_model(next_obs_batch).detach().max(1)[0]

    # 这个公式基于贝尔曼方程（思路和动态规划一样）
    # 这个公式表明，一个动作的预期 Q 值等于即时奖励加上在下一个状态下所有可能动作的最大预期 Q 值的折扣值。这个折扣值反映了未来奖励的当前价值。
    expected_Q_values = reward_batch + (gamma * next_Q_values * (1 - done_batch))

    # 为什么要乘以 (1 - done_batch)
    # 当一个回合结束时（例如智能体到达了目标状态或触发了游戏结束的条件），在该状态下没有未来的奖励。
    # 乘以 (1 - done_batch) 确保了如果回合结束，未来奖励的贡献为零。
    # 换句话说，如果 done_batch 中的值为 1（表示回合结束），next_Q_values 将不会对计算的 expected_Q_values 产生影响。
    return expected_Q_values


def optimize_params(model, optimizer, loss) :
    # 优化模型
    optimizer.zero_grad() # 清除之前的梯度。PyTorch 会默认累积梯度，如果不清除梯度，新计算的梯度会被加到已存在的梯度上，在 DQN 中这会使得训练过程变得不稳定，甚至可能导致模型完全无法学习。
    loss.backward()             # 反向传播，计算梯度
    # 梯度裁剪，防止梯度爆炸
    # 梯度爆炸会导致模型权重的大幅更新，使得模型无法收敛或表现出不稳定的行为
    for param in model.parameters() :
        param.grad.data.clamp_(-1, 1)   # 限制梯度值在 -1 到 1 的范围内，这是防止梯度值变得过大或过小、导致训练不稳定
    optimizer.step()      # 更新参数（梯度下降，指使用计算出的梯度来更新模型参数的过程）



if __name__ == '__main__' :
    main(arguments())
