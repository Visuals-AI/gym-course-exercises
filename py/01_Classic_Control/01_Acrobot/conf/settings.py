#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------


# 检查点配置
CHECKPOINTS_DIR = './out/checkpoints'
CHECKPOINT_PREFIX = 'checkpoint_epoch'
CHECKPOINT_SUFFIX = '.pth'
SAVE_CHECKPOINT_INTERVAL = 10


# 训练完成后的模型输出位置
MODELS_DIR = './out/models'
ACROBOT_MODEL_NAME = 'acrobot_model.pth'
ACROBOT_MODEL_PATH = f"{MODELS_DIR}/{ACROBOT_MODEL_NAME}"


