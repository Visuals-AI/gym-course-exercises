#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/17 20:33
# -----------------------------------------------

# 课程名字
COURSE_NAME = 'mountain_car_continuous'

# 训练的模型名称
MODEL_NAME = 'mountain_car_continuous'
ACTOR_MODEL_NAME = 'actor'
CRITIC_MODEL_NAME = 'critic'

# 交互的环境名称
ENV_NAME = 'MountainCarContinuous-v0'
MAX_STEP = 999  # MountainCarContinuous 问题的 v0 版本要求在 999 步内完成


# 引入公共配置项
from conf.global_settings import *
