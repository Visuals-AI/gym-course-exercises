#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------

import os
import time
import torch


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)



def current_millis() :
    return int(time.time() * 1000)



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

