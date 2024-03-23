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

# 添加公共库文件的相对位置
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../libs/')
# --------------------

import re
import argparse
import glob
import torch
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
    parser.add_argument('-u', '--human', dest='human', action='store_true', default=False, help='渲染模式: 人类模式，帧率较低且无法更改窗体显示内容')
    parser.add_argument('-a', '--rgb_array', dest='rgb_array', action='store_true', default=False, help='渲染模式: RGB 数组，需要用 OpenCV 等库辅助渲染，可以在每一帧添加定制内容，帧率较高')
    parser.add_argument('-s', '--save_gif', dest='save_gif', action='store_true', default=False, help='保存每个回合渲染的 UI 到 GIF（仅 rgb_array 模式有效）')
    parser.add_argument('-m', '--model', dest='model', type=str, default='', help='验证单个模型的路径；如果为空，则验证所有模型')
    parser.add_argument('-c', '--cpu', dest='cpu', action='store_true', default=False, help='强制使用 CPU: 默认情况下，自动优先使用 GPU 训练（除非没有 GPU）')
    parser.add_argument('-e', '--epoches', dest='epoches', type=int, default=100, help='验证次数')
    return parser.parse_args()


def main(args) :
    if args.model :
        test_model(args.model, args)

    else :
        test_models(args)


def test_models(args) :
    model_dir = os.path.dirname(get_model_path(COURSE_NAME, MODEL_NAME))
    path_pattern = os.path.join(model_dir, f"*{MODEL_SUFFIX}")
    model_paths = glob.glob(path_pattern)

    # 验证每个模型的成功率
    percentages = {}
    for model_path in model_paths :
        percentage = test_model(model_path, args)
        percentages[model_path] = percentage

    # 找出成功率最好的模型（不是训练次数越多就多好的，有可能存在过拟合问题）
    log.info("各个模型的验证如下:")
    optimal_model_path = ''
    max_percentage = 0
    sorted_model_paths = sorted(percentages, key=extract_number)
    for model_path in sorted_model_paths :
        percentage = percentages.get(model_path) or 0
        log.info(f"  模型 [{os.path.basename(model_path)}] 挑战成功率为: [{percentage:.2f}%]")
        if max_percentage < percentage :
            max_percentage = percentage
            optimal_model_path = model_path
    log.warn(f"最优模型为: [{optimal_model_path}]")
    log.warn(f"挑战成功率为: [{max_percentage:.2f}%]")

    

def test_model(model_path, args) :
    '''
    加载训练好的模型，重复验证，计算通过率。
    :params: model_path 待验证的模型路径
    :params: args 从命令行传入的训练控制参数
    :return: None
    '''
    # 设置为评估模式
    targs = TrainArgs(args, eval=True)
    
    # 加载模型参数  
    targs.model.load_state_dict(         
        torch.load(model_path)
    )
    
    log.info("++++++++++++++++++++++++++++++++++++++++")
    log.info(f"开始验证模型: {model_path}")
    cnt_ok = 0
    min_step = MAX_STEP
    max_step = 0
    avg_step = 0
    for epoch in range(1, args.epoches + 1) :
        log.debug(f"第 {epoch}/{args.epoches} 回合验证开始 ...")
        step = test(targs, epoch)
        is_ok = (step < MAX_STEP)
        cnt_ok += (1 if is_ok else 0)

        min_step = (min_step if min_step < step else step)
        max_step = (max_step if max_step > step else step)
        avg_step += step

    avg_step = int(avg_step / args.epoches)
    percentage = (cnt_ok / args.epoches) * 100
    log.warn(f"已完成模型 [{os.path.basename(model_path)}] 的验证，挑战成功率为: {percentage:.2f}%")
    log.warn(f"本次验证中，智能体完成挑战的最小步数为 [{min_step}], 最大步数为 [{max_step}], 平均步数为 [{avg_step}]")
    log.info("----------------------------------------")
    targs.close_env()
    return percentage


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
    cnt_step = 0
    for _ in range(MAX_STEP) :

        # 选择下一步动作
        action = select_next_action(targs.model, obs)
        
        # 执行动作并获取下一个状态
        next_obs, _, done, _, _ = targs.env.step(action)
        obs = to_tensor(next_obs, targs)

        # 渲染 GUI
        cnt_step +=1
        labels = [
            f"epoch: {epoch}", 
            f"step: {cnt_step}", 
            f"action: {action}", 
            f"cos01: {obs[0][0]}",              # 第一个关节角度的余弦值（这个角度是指第一个关节与垂直向下位置的夹角）
            f"sin01: {obs[0][1]}",              # 第一个关节角度的正弦值
            f"cos02: {obs[0][2]}",              # 第二个关节角度的余弦值（这个角度是相对于第一个关节的）
            f"sin02: {obs[0][3]}",              # 第二个关节角度的正弦值
            f"angular_velocity1: {obs[0][4]}",  # 第一个关节的角速度
            f"angular_velocity2: {obs[0][5]}",  # 第二个关节的角速度
        ]
        targs.render(labels)

        # log.debug(f"[第 {epoch} 回合] 已执行 {cnt_step} 步: {action}")
        if done :
            break
    
    # 保存智能体这个回合渲染的动作 UI
    targs.save_render_ui(epoch)
        
    if cnt_step < MAX_STEP :
        log.debug(f"[第 {epoch} 回合] 智能体在第 {cnt_step} 步完成挑战")
    else :
        log.debug(f"[第 {epoch} 回合] 智能体未能在 {MAX_STEP} 步内完成挑战")
    return cnt_step


def select_next_action(model, obs) :
    '''
    使用模型推理下一步的动作
    :params: model 被测模型
    :params: obs 当前观察空间
    :return: 下一步动作
    '''
    with torch.no_grad() :  # no_grad 告诉 PyTorch 在这个块中不要计算梯度。
                            # 在推理过程中，是使用模型来预测输出，而不是通过反向传播来更新模型的权重。
        
        # 传递输入数据到模型
        model_output = model(obs)

        # 在模型的输出中找到具有最大 Q 值的动作的索引
        action_index = model_output.max(1)[1]

        # 调整张量形状为 (1, 1)
        action_index_reshaped = action_index.view(1, 1)

        # 获取单个动作值
        action = action_index_reshaped.item()
    return action


def extract_number(filepath) :
    '''
    自定义排序函数
    :params: filepath 文件路径
    :return: 文件名字符串的排序模式
    '''
    filename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0


if __name__ == '__main__' :
    main(arguments())
