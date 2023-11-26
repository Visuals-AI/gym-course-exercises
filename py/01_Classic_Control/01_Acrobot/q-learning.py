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


import numpy as np
import gymnasium as gym
from color_log.clog import log


def main() :
    # 创建和配置环境
    # ------
    # Acrobot-v1 表示使用 Acrobot（版本v1）的预设环境
    # 预设环境的好处是不需要我们定义 “行动空间” 、“观察空间/状态空间”、“奖励系统”、“交互接口” 等等，
    # 我们只需要关注 “训练算法” 本身，预设环境有一个统一标准、方便我们入门学习和交流
    # ------
    # 但是换言之，譬如以后我们要训练某个游戏的强化学习模型，除了算法之外，我们还要自己定义环境
    env = gym.make('Acrobot-v1')

    # 实现 “训练算法” 以进行训练
    # 针对 Acrobot 问题， Q-learning 算法会更适合：
    #   Q-learning（贪婪策略）：
    #       在更新值函数时，Q-learning 考虑的是下一个状态中可能获得的最大回报。
    #       它使用贪婪策略来更新 Q 值，即选择下一个状态中 Q 值最大的动作来进行更新。
    #       换言之，它侧重于找到最优解并且在理论上能够更快地收敛到最佳策略。
    #       Acrobot 问题需要找到快速摆动到目标位置的策略，Q-learning 的贪婪特性可能在这方面表现更好。
    train_by_qlearning(env)


def train_by_qlearning(env) :
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

    # ------------------------------------------
    # 初始化 Q 表
    # 由于 Acrobot 的状态空间较大，这里简化为一个较小的表
    q_table = np.zeros(     # 创建一个全 0 数组
        [
            env.observation_space.shape[0],     # 获取 Acrobot 环境的状态空间维度（即 6）。对于 Acrobot，状态是一个多维的数据，shape[0] 获取这个状态空间的第一个维度的大小
            env.action_space.n                  # 获取 Acrobot 环境的动作空间中、可用动作的数量（即 3）
        ]
    )   # 创建了一个 6x3 的全 0 二维数组，行数对于状态空间的维度、列数对于动作空间的大小。该表用于存储和更新每个 “状态-动作” 的 Q 值

    # 设置学习参数（Q-learning 的关键参数）
    alpha = 0.1     # Alpha (α) - 学习率，或步长：它决定了新信息覆盖旧信息的程度。较高的值意味着学习快速，但可能导致学习过程不稳定或收敛到次优策略。较低的值意味着学习慢，但能增加稳定性。通常设置在 0.1 到 0.5 之间。这里 0.1 是一个保守的选择，确保学习过程稳定。
    gamma = 0.99    # Gamma (γ) - 折扣因子：用于计算未来奖励的当前价值。它决定了未来奖励对当前决策的影响程度。值越高，智能体越重视长远利益。通常设置在 0.9 到 0.99 之间。这里 0.99 的值表明智能体在做出决策时非常重视未来的奖励。
    epsilon = 0.1   # Epsilon (ε) - 探索率：用于 epsilon-greedy 策略，它决定了智能体探索新动作的频率。值越高，智能体越倾向于尝试新的、不确定的动作而不是已知的最佳动作。这个值通常在训练初期较高，以鼓励探索，随着学习的进行逐渐降低。这里 0.1 是一个相对较低的值，表示智能体在大部分时间会采取当前认为最好的动作。


    # ------------------------------------------
    # 训练循环
    log.info("开始训练 ...")
    for i in range(1000):
        state = env.reset()     # 重置 Acrobot 状态
        done = False
        log.info("++++++++++++++++++++++++++++++++++++++++")
        log.info(f"第 {i} 轮训练开始 ...")

        while not done:
            # epsilon-greedy 策略
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # 随机选择一个动作
            else:
                action = np.argmax(q_table[state])  # 选择 Q 值最高的动作
            log.debug("执行动作：{action}")

            # 执行动作（与环境交互）并观察结果
            next_state, reward, done, _ = env.step(action)
            log.debug("执行结果：")
            log.debug("  next_state 状态空间变化：{next_state}")    # 执行动作后的新状态或观察。这是智能体在下一个时间步将观察到的环境状态。
            log.debug("  reward 获得奖励情况： {reward}")           # 执行动作后获得的奖励。这是一个数值，指示执行该动作的效果好坏，是强化学习中的关键信号。
            log.debug("  done 当前回合是否结束: {done}")            # 可能成功也可能失败，例如在一些游戏中，达到目标或失败会结束回合。
            # log.debug("  额外信息: {_}")                          # 通常用 hash 表附带自定义的额外信息（如诊断信息、调试信息），这里用下划线来表示不需要用到的额外信息。

            # 更新 Q 表
            old_value = q_table[state, action]      # 获取当前 “状态-动作” 之前学到的价值估计
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            state = next_state

        log.info(f"第 {i} 轮训练结束")
        log.info("----------------------------------------")

    # 关闭环境
    env.close()
    log.info("训练结束")


if __name__ == '__main__' :
    main()
