#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------

import os
import time
import glob
import torch
import numpy as np
from conf.global_settings import MODEL_SUFFIX


def create_dirs(path):
    '''
    创建目录（支持多层）
    :params: path 目录路径
    :return: None
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def remove_dirs(path):
    '''
    删除目录（支持多层）
    :params: path 目录路径
    :return: None
    '''
    if not os.path.exists(path):
        os.removedirs(path)


def get_model_group_paths(model_dir, model_epoch) :
    '''
    根据回合数从模型目录中选择同一组模型路径
    :params: model_epoch 训练模型的回合数
    :return: 同一组模型路径
    '''
    path_pattern = os.path.join(model_dir, f"*{model_epoch}{MODEL_SUFFIX}")
    model_paths = glob.glob(path_pattern)
    return model_paths


def is_close_to_zero(num, threshold=1e-4):
    return abs(num) < threshold


def current_millis() :
    '''
    当前时间戳（毫秒级）
    :return: 当前时间戳（毫秒级）
    '''
    return int(time.time() * 1000)


def current_seconds() :
    '''
    当前时间戳（秒级）
    :return: 当前时间戳（秒级）
    '''
    return int(time.time())



def scan_device(use_cpu=False) :
    '''
    扫描可用设备。
    默认情况下，如果同时存在 GPU 和 CPU，优先使用 GPU。
    :params: use_cpu 强制使用 CPU
    :return: 可用设备
    '''
    device_name = "cuda" if not use_cpu and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    return device


def to_tensor(obs_state, targs, batch=False) :
    '''
    把观测空间的当前状态转换为 PyTorch 张量并送入神经网络
    :params: obs_state 观测空间的当前状态
    :params: targs 训练参数
    :return: 观测空间
    '''
    if not batch:
        # 对单个观测状态进行重塑
        # 把观察空间的状态数组转换成 1 x obs_size，目的是确保状态数组与 DQN 神经网络的输入层匹配
        obs_state = np.reshape(obs_state, [1, targs.obs_size])
    else :
        pass # 直接转换批量数据，不进行重塑

    # 把观察空间的状态数组送入神经网络所在的设备   
    obs_state = _to_tensor(obs_state, targs.device)                  
    return obs_state


def _to_tensor(obs_state, device):
    '''
    把观测空间的当前状态转换为 PyTorch 张量并送入神经网络
    :params: obs_state 观测空间的当前状态
    :params: device 当前运行神经网络的设备： GPU/CPU
    :return: 观测空间
    '''
    if isinstance(obs_state, np.ndarray):
        return torch.from_numpy(obs_state).float().to(device)
    
    elif isinstance(obs_state, torch.Tensor):
        return obs_state.to(device)
    
    else:
        return obs_state



def to_nparray(act_state) :
    '''
    把动作空间的当前状态转换为 Numpy 张量并送入神经网络
    :params: act_state 动作空间的当前状态
    :return: 动作空间数组
    '''
    if isinstance(act_state, torch.Tensor):
        action = act_state.numpy().flatten()
    else :
        action = act_state
    return action


def up_dim(low_tensor) :
    '''
    对张量升维。

    如 tensor([1 2 3 4]) 长度为 4 的一维张量，
    会升维为形状为 4x1 的二维张量 （4 行 1 列）
        tensor(
            [
                [1] 
                [2] 
                [3] 
                [4]
            ]
        )
    :params: low_tensor 低维张量
    :return: 高维张量
    '''
    return low_tensor.unsqueeze(-1)


def down_dim(high_tensor) :
    '''
    对张量降维。
    
    如形状为 4x1 的二维张量 （4 行 1 列）
        tensor(
            [
                [1] 
                [2] 
                [3] 
                [4]
            ]
        )
    会降维成 tensor([1 2 3 4]) 长度为 4 的一维张量
    :params: high_tensor 高维张量
    :return: 高维张量
    '''
    return high_tensor.squeeze(1)



def find_optimal_result(tested_results, min_best=True) :
    '''
    寻找最优的测试结果
    :params: tested_results 测试结果集
    :params: min_best 最小步数为最优; 反之则最大步数为最优
    :return: 高维张量
    '''
    tested_results.sort(key=lambda x: x.percentage, reverse=True)

    # 然后，找到具有相同最大 percentage 的所有结果
    max_percentage = tested_results[0].percentage
    candidates = [result for result in tested_results if result.percentage == max_percentage]

    # 最小步数 为最优
    if min_best :
        optimal_result = min(candidates, key=lambda x: x.min_step)

    # 最大步数为最优
    else :
        optimal_result = max(candidates, key=lambda x: x.max_step)

    return optimal_result
