#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------

import os
import time
import torch
import numpy as np


def create_dirs(path):
    '''
    创建目录（支持多层）
    :params: path 目录路径
    :return: None
    '''
    if not os.path.exists(path):
        os.makedirs(path)



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

