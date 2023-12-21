#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------


# 训练一定次数后的模型输出位置
MODELS_DIR = './out/models'
MODEL_SUFFIX = '.pth'
def get_model_path(model_name, epoch=0) :
    return f"{MODELS_DIR}/{model_name}/{model_name}_model_epoch_{epoch}{MODEL_SUFFIX}"


# 检查点配置
CHECKPOINTS_DIR = './out/checkpoints'
CHECKPOINT_SUFFIX = '.pth'
SAVE_INTERVAL = 500
def get_checkpoint_path(model_name, epoch=0) :
    return f"{CHECKPOINTS_DIR}/{model_name}/{model_name}_model_epoch_{epoch}{CHECKPOINT_SUFFIX}"



# 训练过程日志数据
TENSOR_DIR = './out/tensor'
def get_tensor_path(model_name) :
    return f"{TENSOR_DIR}/{model_name}"

