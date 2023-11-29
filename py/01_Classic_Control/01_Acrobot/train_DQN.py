#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------
# 经典控制： Acrobot （杂技机器人）
#   Acrobot 是一个双节摆问题，目标是用最少的步骤使得摆的末端达到一定高度。
# 
# 相关文档：
#   https://gymnasium.farama.org/environments/classic_control/acrobot/
#   http://incompleteideas.net/book/11/node4.html
# -----------------------------------------------


import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
from tensorboardX import SummaryWriter
from bean.dqn import DQN
import random
import numpy as np
from collections import deque
import gymnasium as gym
from bean.checkpoint import CheckpointManager
from tools.utils import *
from conf.settings import *
from color_log.clog import log


def arguments() :
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='Gym - Acrobot 训练脚本',
        description='在默认环境下、使用深度 Q 网络（DQN）训练智能体操作 Acrobot', 
        epilog='\r\n'.join([
            '运行环境: python3', 
            '运行示例: python py/01_Classic_Control/01_Acrobot/train_DQN.py'
        ])
    )
    parser.add_argument('-r', '--render', dest='render', action='store_true', default=False, help='渲染模式: 可以通过 GUI 观察智能体实时交互情况')
    parser.add_argument('-c', '--cpu', dest='cpu', action='store_true', default=False, help='强制使用 CPU: 默认情况下，自动优先使用 GPU 训练（除非没有 GPU）')
    parser.add_argument('-e', '--episodes', dest='episodes', type=int, default=1000, help='训练次数: 即训练过程中智能体将经历的总回合数。每个回合是一个从初始状态到终止状态的完整序列')
    parser.add_argument('-g', '--gamma', dest='gamma', type=float, default=0.95, help='折扣因子: 用于折算未来奖励的在当前回合中的价值。它决定了未来奖励对当前决策的影响程度。值越高，智能体越重视长远利益，通常设置在 [0.9, 0.99]')
    parser.add_argument('-s', '--epsilon', dest='epsilon', type=float, default=1.0, help='探索率: 用于 epsilon-greedy 策略，它决定了智能体探索新动作的频率。值越高，智能体越倾向于尝试新的、不确定的动作而不是已知的最佳动作。这个值通常在训练初期较高，随着学习的进行逐渐降低')
    parser.add_argument('-d', '--epsilon_decay', dest='epsilon_decay', type=float, default=0.995, help='衰减率: 探索率随时间逐渐减小的速率。每经过一个回合，epsilon 将乘以这个衰减率，从而随着时间的推移减少随机探索的频率')
    parser.add_argument('-m', '--min_epsilon', dest='min_epsilon', type=float, default=0.1, help='最小探索率: 即使经过多次衰减，探索率也不会低于这个值，确保了即使在后期也有一定程度的探索')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32, help='从经验回放存储中一次抽取并用于训练网络的经验的数量。默认为 32，意味着每次训练时会使用 32 个经验样本')
    return parser.parse_args()


def main(args) :
    # 创建和配置环境
    # ------
    # Acrobot-v1 表示使用 Acrobot（版本v1）的预设环境
    # 预设环境的好处是不需要我们定义 “行动空间” 、“观察空间/状态空间”、“奖励系统”、“交互接口” 等等，
    # 我们只需要关注 “训练算法” 本身，预设环境有一个统一标准、方便我们入门学习和交流
    # ------
    # 但是换言之，譬如以后我们要训练某个游戏的强化学习模型，除了算法之外，我们还要自己定义环境
    env = gym.make('Acrobot-v1', render_mode="human")

    # 从 Acrobot 文档中可知 状态空间（或观察空间） observation_space = Box([ -1. -1. -1. -1. -12.566371 -28.274334], [ 1. 1. 1. 1. 12.566371 28.274334], (6,), float32)
    #   Box 是 gym 定义的数据类型，代表一个 n 维的盒子，可以用来定义在每个维度上的连续值范围: 
    #       [ -1. -1. -1. -1. -12.566371 -28.274334] 是观察空间中每个维度的最小值
    #       [ 1. 1. 1. 1. 12.566371 28.274334] 是观察空间中每个维度的最大值
    #       (6,)  表示观察空间是一个六维的空间。这是一个元组，其中只有一个元素，即 6，表明有六个独立的观察值。
    #       float32 表示这些值是 32 位浮点数
    #   
    #   Acrobot 的六个独立的观察值分别代表：
    #       Cosine of theta1：第一个关节角度的余弦值。这个角度是指第一个链接与垂直向下位置的夹角。
    #       Sine of theta1：第一个关节角度的正弦值。
    #       Cosine of theta2：第二个关节角度的余弦值。这个角度是相对于第一个链接的。
    #       Sine of theta2：第二个关节角度的正弦值。
    #       Angular velocity of theta1：第一个关节的角速度。
    #       Angular velocity of theta2：第二个关节的角速度。
    log.debug(env.observation_space)

    # 从 Acrobot 文档中可知 动作空间 env.action_space 只有 3 个值，
    # 它代表了可以施加在两个链接之间的活动关节上的力矩（torque）的大小：
    #   动作 0：施加 -1 的力矩到活动关节上。
    #   动作 1：施加 0 的力矩到活动关节上。
    #   动作 2：施加 1 的力矩到活动关节上。
    # 这些动作是离散的，通过选择不同的动作，智能体可以控制 Acrobot 的行动，使其实现特定的运动目标，如摆动到一定高度。
    log.debug(env.action_space)

    # 实现 “训练算法” 以进行训练
    # 针对 Acrobot 问题， DQN 算法会更适合：
    #   DQN（Deep Q-Network）是一种将深度学习与强化学习相结合的算法
    #   它主要用于解决具有连续、高维状态空间的问题，特别是那些传统的 Q-learning 算法难以处理的问题。
    #   在 DQN 中，传统 Q-learning 中的 Q 表（一个用于存储所有状态-动作对应价值的巨大表格）被一个深度神经网络所替代。
    #   这个神经网络被训练来预测给定状态和动作下的 Q 值
    train_dqn(args, env)


def train_dqn(args, env) :
    # TODO： TensorBoard 怎么看
    writer = SummaryWriter()

    # ------------------------------------------
    # 深度 Q 网络（DQN）算法的关键参数和设置
    state_size = env.observation_space.shape[0]     # Acrobot 状态空间维度
    action_size = env.action_space.n                # Acrobot 动作空间数量

    model = DQN(state_size, action_size)  # DQN 简单的三层网络模型
    memory = deque(maxlen=2000)           # 创建一个双端队列（deque），作为经验回放的存储。当存储超过2000个元素时，最旧的元素将被移除。经验回放是DQN中的一项关键技术，有助于打破经验间的相关性并提高学习的效率和稳定性。

    # ------------------------------------------
    # 检查 GPU 是否可用
    device = scan_device(args.cpu)
    model.to(device)    # 将模型和优化器移动到 GPU


    optimizer = optim.Adam(model.parameters(), lr=0.001)    # 定义了用于训练神经网络的优化器。这里使用的是Adam优化器，一个流行的梯度下降变种，lr=0.001设置了学习率为0.001。
    criterion = nn.MSELoss()    # 这定义了用于训练过程中的损失函数。这里使用的是均方误差损失（MSE Loss），它是评估神经网络预测值与实际值差异的常用方法。


    # ------------------------------------------
    cur_episode = 0
    epsilon = args.epsilon

    checkpoint_manager = CheckpointManager()
    last_checkpoint = checkpoint_manager.load_last_checkpoint()
    if last_checkpoint :
        cur_episode = last_checkpoint.episode + 1
        epsilon = last_checkpoint.epsilon
        model.load_state_dict(last_checkpoint.model_state_dict)
        optimizer.load_state_dict(last_checkpoint.optimizer_state_dict)


    # ------------------------------------------
    # 训练循环
    log.info("++++++++++++++++++++++++++++++++++++++++")
    log.info("开始训练 ...")
    for episode in range(cur_episode, args.episodes) :
        log.info(f"第 {episode} 轮训练开始 ...")

        state = env.reset()     # 重置环境（在Acrobot环境中，这个初始状态是一个包含了关于Acrobot状态的数组，例如两个连杆的角度和角速度。）
        # state = (array([ 0.9996459 ,  0.02661069,  0.9958208 ,  0.09132832, -0.04581745, -0.06583451], dtype=float32), {})
        state = np.reshape(state[0], [1, state_size])  # 把状态数组转换成 1 x state_size 的数组，为了确保状态数组与神经网络的输入层匹配。
        state = to_tensor(state, device)
        total_reward = 0        # 用于累计智能体从环境中获得的总奖励。在每个训练回合结束时，total_reward将反映智能体在该回合中的总体表现。奖励越高，意味着智能体的性能越好。

        env.render()

        # 添加以下两行，用于记录训练过程信息到TensorBoard
        episode_summary = {'Total Reward': 0, 'Epsilon': epsilon}

        bgn_time = current_millis()
        step_counter = 0
        while True:

            # epsilon-greedy策略：
            # 即智能体在选择动作时，有一定概率随机探索环境，而其余时间则根据已学习的策略选择最佳动作
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()  # 随机选择一个动作

            # 智能体会根据当前经验、选择当前估计最优的动作
            else:
                action = torch.argmax(model(state)).item()

                # action = torch.argmax(                  # 2. 神经网络模型输出每个可能动作的预期Q值。然后，使用torch.argmax选择具有最高预期Q值的动作，这代表了当前状态下的最佳动作。
                #     model(
                #         torch.from_numpy(state).float() # 1. 将当前状态转换为PyTorch张量并传递给神经网络模型（model）
                #     )
                # ).item()                                # 3. item 从张量中提取动作值

            next_state, reward, terminated, truncated, info  = env.step(action)
            # FIXME truncated 状态有问题，需要对于调整 reward
            done = terminated or truncated      # 解释详见 https://stackoverflow.com/questions/73195438/openai-gyms-env-step-what-are-the-values
            # log.debug("执行结果：")
            # log.debug(f"  next_state 状态空间变化：{next_state}")    # 执行动作后的新状态或观察。这是智能体在下一个时间步将观察到的环境状态。
            # log.debug(f"  reward 获得奖励情况： {reward}")           # 执行动作后获得的奖励。这是一个数值，指示执行该动作的效果好坏，是强化学习中的关键信号，作为当次动作的反馈。
            # log.debug(f"  done 当前回合是否结束: {done}")            # 可能成功也可能失败，例如在一些游戏中，达到目标或失败会结束回合。
            # log.debug(f"  info 额外信息: {info}")                   # 通常用 hash 表附带自定义的额外信息（如诊断信息、调试信息），暂时不需要用到的额外信息。

            next_state = np.reshape(next_state, [1, state_size])
            next_state = to_tensor(next_state, device)
            
            memory.append((state, action, reward, next_state, done))    # 向memory（经验回放缓冲区）添加当前 step 执行前后状态、奖励情况等
            state = next_state      # 更新当前状态
            total_reward += reward  # 累计奖励

            if done:
                log.debug(f"Episode: {episode+1}, Total reward: {total_reward}, Epsilon: {epsilon}")

                # 记录本回合的总奖励到TensorBoard
                episode_summary['Total Reward'] = total_reward
                writer.add_scalars('Training', episode_summary, episode)
                break

            # 确保只有当经验回放缓冲区（memory）中的样本数量超过批处理大小（batch_size）时，才进行学习过程。
            # 这是为了确保有足够的样本来进行有效的批量学习。
            # 这个过程是DQN学习算法的核心，它利用从环境中收集的经验（通过经验回放）来不断调整和优化网络，使得预测的Q值尽可能接近实际的Q值。
            # 这种基于值的强化学习方法通过迭代这个过程，逐渐学习到一个策略，该策略可以最大化累积奖励。
            if len(memory) > args.batch_size:
                minibatch = random.sample(memory, args.batch_size)   # 从memory中随机抽取batch_size数量的样本。这种随机抽样是为了减少样本间的相关性，增强学习的稳定性和效率。
                for m_state, m_action, m_reward, m_next_state, m_done in minibatch:

                    # target 即为目标 Q 值。 初始化为观测到的即时奖励（m_reward）
                    target = m_reward

                    # 如果回合尚未结束，则计算目标Q值。
                    if not m_done:
                        # 使用Bellman方程计算目标Q值。
                        # 这涉及到将下一个状态（m_next_state）输入到网络中，以估计在该状态下所有可能动作的最大Q值，
                        # 然后将这个最大Q值乘以折扣因子（gamma）并加上即时奖励（m_reward）
                        # target = m_reward + gamma * torch.max(model(torch.from_numpy(m_next_state).float())).item()
                        target = m_reward + args.gamma * torch.max(model(m_next_state)).item()

                        # 简单说明 Bellman 方程，它是 动态规划 中的一个概念： 
                        #   一个状态的最优价值是在该状态下所有可能动作中可以获得的最大回报，其中每个动作的回报是即时奖励加上下一个状态在最优策略下的折扣后的价值。

                    
                    target_f = model(m_state) # 通过神经网络模型预测当前状态（m_state）下的 Q 值。
                    target_f[0][m_action] = target                      # 更新与执行的动作（m_action）对应的 Q 值为之前计算的目标 Q 值。
                    optimizer.zero_grad()                               # 在每次网络更新前清除旧的梯度，这是PyTorch的标准做法。
                    loss = criterion(target_f, model(m_state))    # 计算预测 Q 值（target_f）和通过网络重新预测当前状态 Q 值之间的损失。
                    loss.backward()         # 对损失进行反向传播，计算梯度
                    optimizer.step()        # 根据计算的梯度更新网络参数

                    step_counter += 1

        end_time = current_millis()
        # while end

        # 在训练循环外，添加以下两行，用于记录每个回合结束时的步数到TensorBoard
        writer.add_scalar('Training/Steps per Episode', step_counter, episode)

        # ε-贪婪策略（epsilon-greedy strategy）的强化学习技巧中的关键部分，用于平衡探索（exploration）和利用（exploitation）
        #   在强化学习中，智能体需要决定是利用当前已知的最佳策略（exploitation）来最大化短期奖励，还是探索新的动作（exploration）以获得更多信息，可能会带来更大的长期利益。
        #   ε-贪婪策略通过一个参数ε（epsilon）来控制这种平衡。ε的值是一个0到1之间的数字，表示选择随机探索的概率。
        epsilon = max(args.min_epsilon, args.epsilon_decay * epsilon) # 衰减探索率

        checkpoint_manager.save_checkpoint(model, optimizer, episode, epsilon)
        log.info(f"第 {episode} 轮训练结束")
    # for end

    # 关闭TensorBoard的SummaryWriter
    writer.close()

    env.close()
    log.info("训练结束")

    torch.save(model.state_dict(), ACROBOT_MODEL_PATH)
    log.info(f"模型已保存到 {ACROBOT_MODEL_PATH}")
    log.info("----------------------------------------")



def scan_device(use_cpu=False) :
    '''
    扫描可用设备。
    默认情况下，如果同时存在 GPU 和 CPU，优先使用 GPU。
    params: use_cpu 强制使用 CPU
    return: 可用设备
    '''
    device_name = "cuda" if not use_cpu and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    return device


def to_tensor(data, device):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


if __name__ == '__main__' :
    main(arguments())
