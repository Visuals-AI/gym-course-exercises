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
    if not os.path.exists(path):
        os.makedirs(path)



def current_millis() :
    return int(time.time() * 1000)



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



def to_tensor(obs_state, device):
    '''
    把观测空间的当前状态转换为 PyTorch 张量并送入神经网络
    :params: obs_state 观测空间的当前状态
    :params: device 当前运行神经网络的设备： GPU/CPU
    :return: 
    '''
    if isinstance(obs_state, np.ndarray):
        return torch.from_numpy(obs_state).float().to(device)
    
    elif isinstance(obs_state, torch.Tensor):
        return obs_state.to(device)
    
    else:
        return obs_state
