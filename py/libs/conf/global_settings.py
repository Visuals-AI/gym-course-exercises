#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------


# 训练一定次数后的模型输出位置
MODELS_DIR = './out/models'
MODEL_SUFFIX = '.pth'
def get_model_path(course_name, model_name, epoch=0) :
    return f"{MODELS_DIR}/{course_name}/{model_name}_model_epoch_{epoch}{MODEL_SUFFIX}"


# 检查点配置
CHECKPOINTS_DIR = './out/checkpoints'
CHECKPOINT_SUFFIX = '.pth'
SAVE_INTERVAL = 200
def get_checkpoint_path(course_name, model_name, epoch=0) :
    return f"{CHECKPOINTS_DIR}/{course_name}/{model_name}_model_epoch_{epoch}{CHECKPOINT_SUFFIX}"



# 训练过程日志数据
TENSOR_DIR = './out/tensor'
def get_tensor_path(course_name) :
    return f"{TENSOR_DIR}/{course_name}"



# 渲染 UI 保存位置
RENDER_UI_DIR = './out/ui'
def get_render_ui_path(course_name, model_name, epoch=0, eval=False) :
    subdir = "test" if eval else "train"
    return f"{RENDER_UI_DIR}/{course_name}/{subdir}/{model_name}_ui_epoch_{epoch}.gif"
    
