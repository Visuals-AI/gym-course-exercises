#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------


# 训练的模型名称
MODEL_NAME = 'mountain_car_continuous'

# 交互的环境名称
ENV_NAME = 'MountainCarContinuous-v0'
MAX_STEP = 999  # MountainCarContinuous 问题的 v0 版本要求在 999 步内完成


# 引入公共配置项
from conf.global_settings import *
