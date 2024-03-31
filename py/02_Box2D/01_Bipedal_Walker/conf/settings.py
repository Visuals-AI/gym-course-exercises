#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/31 13:52
# -----------------------------------------------

# 课程名字
COURSE_NAME = 'bipedal_walker'

# 训练的模型名称
MODEL_NAME = 'bipedal_walker'
ACTOR_MODEL_NAME = 'actor'
CRITIC_MODEL_NAME = 'critic'

# 交互的环境名称
ENV_NAME = 'BipedalWalker-v3'
MAX_STEP = 2000  # bipedal_walker 问题的 v3 版本要在 2000 步内完成任务


# 引入公共配置项
from conf.global_settings import *
