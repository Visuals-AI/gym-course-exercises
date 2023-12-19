#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------

# 训练的模型名称
MODEL_NAME = 'acrobot'


# 训练一定次数后的模型输出位置
MODELS_DIR = './out/models'
MODEL_SUFFIX = '.pth'
MODEL_PATH_FORMAT = f"{MODELS_DIR}/{MODEL_NAME}/{MODEL_NAME}_model_epoch_%d{MODEL_SUFFIX}"


# 检查点配置
CHECKPOINTS_DIR = './out/checkpoints'
CHECKPOINT_SUFFIX = '.pth'
CHECKPOINT_PATH_FORMAT = f"{CHECKPOINTS_DIR}/{MODEL_NAME}/{MODEL_NAME}_checkpoint_epoch_%d{CHECKPOINT_SUFFIX}"
SAVE_INTERVAL = 10


# 训练过程日志数据
TENSOR_DIR = './out/tensor'
TENSOR_PATH = f"{TENSOR_DIR}/{MODEL_NAME}"
