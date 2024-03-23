#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------

# 课程名字
COURSE_NAME = 'pendulum'

# 训练的模型名称
MODEL_NAME = 'pendulum'
ACTOR_MODEL_NAME = 'actor'
CRITIC_MODEL_NAME = 'critic'

# 交互的环境名称
ENV_NAME = 'Pendulum-v1'
MAX_STEP = 200  # Pendulum 问题的 v1 版本要坚持 200 步


# 引入公共配置项
from conf.global_settings import *
