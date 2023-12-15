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
import gymnasium as gym
from bean.train_args import TrainArgs
from tools.utils import *
from conf.settings import *
from color_log.clog import log


def arguments() :
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='Gym - Acrobot 测试脚本',
        description='在默认环境下、使用深度 Q 网络（DQN）验证智能体操作 Acrobot', 
        epilog='\r\n'.join([
            '运行环境: python3', 
            '运行示例: python py/01_Classic_Control/01_Acrobot/test_DQN.py'
        ])
    )
    parser.add_argument('-r', '--render', dest='render', action='store_true', default=False, help='渲染模式: 可以通过 GUI 观察智能体实时交互情况，但是会极大拉低训练效率')
    parser.add_argument('-c', '--cpu', dest='cpu', action='store_true', default=False, help='强制使用 CPU: 默认情况下，自动优先使用 GPU 训练（除非没有 GPU）')
    parser.add_argument('-e', '--epoches', dest='epoches', type=int, default=100, help='验证次数')
    return parser.parse_args()


def main(args) :
    env = gym.make('Acrobot-v1', 
                    render_mode=("human" if args.render else None)
    )   # 验证时如果有需要，可以渲染 GUI 观察实时挑战情况

    test_model(args, env)
    


def test_model(args, env) :
    '''
    加载训练好的模型，重复验证，计算通过率。
    :params: args 从命令行传入的训练控制参数
    :params: env 当前交互的环境变量，如 Acrobot
    :return: None
    '''

    targs = TrainArgs(args, env, 
                      eval=True     # 设置为评估模式
    )
    
    # 加载模型参数  
    targs.model.load_state_dict(         
        torch.load(ACROBOT_MODEL_PATH)
    )
    
    log.info("++++++++++++++++++++++++++++++++++++++++")
    log.info("开始验证模型 ...")
    cnt = 0
    for epoch in range(args.epoches) :
        log.info(f"第 {epoch}/{args.epoches} 回合验证开始 ...")
        is_ok = test(targs, epoch + 1)
        cnt += (1 if is_ok else 0)

    percentage = (cnt / args.epoches) * 100
    log.warn(f"已完成全部验证，挑战成功率为: {percentage:.2f}%")
    log.info("----------------------------------------")
    env.close()


def test(targs : TrainArgs, epoch) :
    '''
    验证模型是否完成挑战。
    :params: targs 用于运行模型的环境和关键参数
    :params: epoch 正在验证的回合数
    :return: 是否完成挑战
    '''
    raw_obs = targs.env.reset()

    # 把观测空间的初始状态转换为 PyTorch 张量，并送入神经网络所在的设备
    obs = to_tensor(raw_obs[0], targs)


    # Acrobot 问题的 v1 版本要求在 500 步内完成
    ACROBOT_V1_MAX_STEP = 500
    step_counter = 0
    for _ in range(ACROBOT_V1_MAX_STEP) :

        # 渲染 GUI（前提是 env 初始化时使用 human 模式）
        if targs.render :
            targs.env.render()

        # 使用模型推理下一步的动作
        with torch.no_grad() :  # no_grad 告诉 PyTorch 在这个块中不要计算梯度。
                                # 在推理过程中，是使用模型来预测输出，而不是通过反向传播来更新模型的权重。
            
            # 传递输入数据到模型
            model_output = targs.model(obs)

            # 在模型的输出中找到具有最大 Q 值的动作的索引
            action_index = model_output.max(1)[1]

            # 调整张量形状为 (1, 1)
            action_index_reshaped = action_index.view(1, 1)

            # 获取单个动作值
            action = action_index_reshaped.item()
        
        
        # 执行动作并获取下一个状态
        next_obs, _, done, _, _ = targs.env.step(action)
        obs = to_tensor(next_obs, targs)
        if done:
            break

        step_counter +=1
        # log.debug(f"[第 {epoch} 回合] 已执行 {step_counter} 步: {action}")
        

    is_ok = False
    if step_counter < ACROBOT_V1_MAX_STEP :
        log.info(f"[第 {epoch} 回合] 智能体在第 {step_counter} 步完成 Acrobot 挑战")
        is_ok = True
    else :
        log.warn(f"[第 {epoch} 回合] 智能体未能在 {ACROBOT_V1_MAX_STEP} 步内完成 Acrobot 挑战")
    return is_ok



if __name__ == '__main__' :
    main(arguments())
