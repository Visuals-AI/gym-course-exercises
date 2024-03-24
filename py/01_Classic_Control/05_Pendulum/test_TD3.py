#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/01/11 13:01
# -----------------------------------------------
# 经典控制： Pendulum （倒立摆-连续动作）
#   Pendulum 是一个倒立摆摆动问题。
#   该系统由一个摆锤组成，摆锤的一端连接到固定点，另一端自由。
#   摆锤从一个随机位置开始，目标是在自由端施加扭矩，将其摆动到直立位置，其重心位于固定点的正上方，然后坚持越久越好。
# 
# 相关文档：
#   https://gymnasium.farama.org/environments/classic_control/pendulum/
# -----------------------------------------------

# 添加公共库文件的相对位置
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../libs/')
# --------------------

import re
import argparse
import torch
from bean.train_args import TrainArgs
from bean.tested_rst import TestedResult
from utils.adjust import *
from tools.utils import *
from conf.settings import *
from color_log.clog import log


def arguments() :
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='Gym - MountainCar 测试脚本',
        description='在默认环境下、使用深度 Q 网络（DQN）验证智能体操作 MountainCar', 
        epilog='\r\n'.join([
            '运行环境: python3', 
            '运行示例: python py/01_Classic_Control/05_Pendulum/test_DQN.py'
        ])
    )
    parser.add_argument('-u', '--human', dest='human', action='store_true', default=False, help='渲染模式: 人类模式，帧率较低且无法更改窗体显示内容')
    parser.add_argument('-a', '--rgb_array', dest='rgb_array', action='store_true', default=False, help='渲染模式: RGB 数组，需要用 OpenCV 等库辅助渲染，可以在每一帧添加定制内容，帧率较高')
    parser.add_argument('-s', '--save_gif', dest='save_gif', action='store_true', default=False, help='保存每个回合渲染的 UI 到 GIF（仅 rgb_array 模式有效）')
    parser.add_argument('-m', '--model_epoch', dest='model_epoch', type=int, default=0, help='根据训练回合数选择验证单个模型')
    parser.add_argument('-c', '--cpu', dest='cpu', action='store_true', default=False, help='强制使用 CPU: 默认情况下，自动优先使用 GPU 训练（除非没有 GPU）')
    parser.add_argument('-e', '--epoches', dest='epoches', type=int, default=100, help='验证次数')
    return parser.parse_args()


def main(args) :
    model_dir = os.path.dirname(get_model_path(COURSE_NAME, MODEL_NAME))
    if args.model_epoch == 0 :
        test_models(args, model_dir)

    else :
        test_model(args, model_dir, args.model_epoch)
        


def test_models(args, model_dir) :

    # 验证每个模型的效果
    tested_rst = {}
    model_epoches = get_model_epoches(model_dir)
    for model_epoch in model_epoches:
        rst = test_model(args, model_dir, model_epoch)
        tested_rst[model_epoch] = rst

    # 找出效果最好的模型（不是训练次数越多就多好的，有可能存在过拟合问题）
    log.info("各个模型的验证如下:")
    sorted_model_epoches = sorted(tested_rst, key=extract_number)
    for model_epoch in sorted_model_epoches :
        rst = tested_rst.get(model_epoch)
        log.info(rst)
        
    optimal_rst = find_optimal_result(list(tested_rst.values()), False)
    log.warn(f"最优模型为: [{optimal_rst.epoch}]")

    

def test_model(args, model_dir, model_epoch) :
    '''
    加载训练好的模型，重复验证，计算通过率。
    :params: args 从命令行传入的训练控制参数
    :params: model_dir 已训练好的模型目录
    :params: model_epoch 模型的训练回合数
    :return: None
    '''
    # 设置为评估模式并加载已训练的模型
    targs = TrainArgs(args, eval=True)
    model_paths = get_model_group_paths(model_dir, model_epoch)    # 通过回合数获取对应一组模型的路径
    targs.load_models(model_paths)
    
    log.info("++++++++++++++++++++++++++++++++++++++++")
    log.info(f"开始验证模型: {model_epoch}")
    cnt_ok = 0
    min_step = MAX_STEP
    max_step = 0
    avg_step = 0
    max_reward = 0
    for epoch in range(1, args.epoches + 1) :
        log.debug(f"第 {epoch}/{args.epoches} 回合验证开始 ...")
        step, reward = test(targs, epoch)
        is_ok = (step >= MAX_STEP)
        cnt_ok += (1 if is_ok else 0)

        max_reward = max(max_reward, reward)
        min_step = min(min_step, step)
        max_step = max(max_step, step)
        avg_step += step

    avg_step = int(avg_step / args.epoches)
    percentage = (cnt_ok / args.epoches) * 100
    log.warn(f"已完成模型 [{model_epoch}] 的验证，挑战成功率为: {percentage:.2f}%")
    log.warn(f"本次验证中，智能体完成挑战的最小步数为 [{min_step}], 最大步数为 [{max_step}], 平均步数为 [{avg_step}]")
    log.info("----------------------------------------")
    targs.close_env()
    return TestedResult(model_epoch, min_step, max_step, avg_step, percentage, max_reward)


def test(targs : TrainArgs, epoch) :
    '''
    验证模型是否完成挑战。
    :params: targs 用于运行模型的环境和关键参数
    :params: epoch 正在验证的回合数
    :return: 是否完成挑战
    '''
    targs.reset_render_cache()
    raw_obs = targs.reset_env()
    obs = to_tensor(raw_obs[0], targs)  # 把观测空间的初始状态转换为 PyTorch 张量，并送入神经网络所在的设备

    # 开始验证
    total_reward = 0
    cnt_step = 0
    for _ in range(MAX_STEP) :

        # 选择下一步动作
        action = select_next_action(targs.actor_model, obs)
        
        # 执行动作并获取下一个状态
        next_obs, reward, done, _, _ = targs.env.step(action)
        obs = to_tensor(next_obs, targs)

        reward = adjust(obs, reward)
        total_reward += reward
        cnt_step +=1

        # 渲染 GUI
        labels = [
            f"epoch: {epoch}", 
            f"step: {cnt_step}", 
            f"action: {action}", 
            f"total_reward: {total_reward}", 
            f"coordinates-x: {obs[0][0]}",      # 自由端的 x 坐标
            f"coordinates-y: {obs[0][1]}",      # 自由端的 y 坐标
            f"angular_velocity: {obs[0][2]}",   # 角速度
        ]
        targs.render(labels)

        # log.debug(f"[第 {epoch} 回合] 已执行 {cnt_step} 步: {action}")
        if done :
            break
    
    # 保存智能体这个回合渲染的动作 UI
    targs.save_render_ui(epoch)
        
    if cnt_step < MAX_STEP :
        log.debug(f"[第 {epoch} 回合] 智能体在第 {cnt_step} 步提前结束挑战")
    else :
        if total_reward > 0 :
            log.debug(f"[第 {epoch} 回合] 智能体挑战坚持 {MAX_STEP} 步成功")
        else :
            log.debug(f"[第 {epoch} 回合] 智能体挑战尝试 {MAX_STEP} 步超时")
            cnt_step = MAX_STEP - 1
    return (cnt_step, total_reward)


def select_next_action(act_model, obs) :
    '''
    使用模型推理下一步的动作。

    TD3 是一个基于 Actor-Critic 框架的算法，
        其中 Actor 负责策略（即选择动作），Critic负责评价这些动作。
    在评估模式下，只需要用 Actor 模型来选择动作，而不需要添加噪声。

    :params: model 被测模型
    :params: obs 当前观察空间
    :return: 下一步动作
    '''
    with torch.no_grad() :  # no_grad 告诉 PyTorch 在这个块中不要计算梯度。
                            # 在推理过程中，是使用模型来预测输出，而不是通过反向传播来更新模型的权重。
        
        # 直接使用 Actor 模型预测动作
        action = act_model(obs).detach()
        
    return to_nparray(action)


def get_model_epoches(model_dir) :
    # 获取已训练好的所有回合数的模型
    model_epoches = []
    filenames = os.listdir(model_dir)
    for filename in filenames:
        match = re.search(r"epoch_(\d+)" + CHECKPOINT_SUFFIX, filename)
        if match:
            epoch = int(match.group(1))
            if epoch not in model_epoches:
                model_epoches.append(epoch)
    return sorted(model_epoches)


def extract_number(model_epoch) :
    '''
    自定义排序函数
    :params: model_epoch 模型的训练回合数
    :return: 
    '''
    return int(model_epoch)



if __name__ == '__main__' :
    main(arguments())
