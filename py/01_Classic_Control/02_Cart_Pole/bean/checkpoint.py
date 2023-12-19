#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------


import re
import os
from datetime import datetime
import torch
from conf.settings import *
from tools.utils import *
from color_log.clog import log


KEY_MODEL_STATE_DICT = 'model_state_dict'
KEY_OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
KEY_EPOCH = 'epoch'
KEY_EPSILON = 'epsilon'
KEY_INFO = 'info'



class Checkpoint :

    def __init__(self, model_state_dict, optimizer_state_dict, epoch, epsilon, info={}) -> None:
        self.model_state_dict = model_state_dict
        self.optimizer_state_dict = optimizer_state_dict
        self.epoch = epoch          # 已训练的回合数（迭代数）
        self.epsilon = epsilon      # 当前探索率
        self.info = info            # 其他附加信息



class CheckpointManager:

    def __init__(self, model_path_format=MODEL_PATH_FORMAT, 
                 checkpoints_path_format=CHECKPOINT_PATH_FORMAT, 
                 save_interval=SAVE_INTERVAL) :
        '''
        初始化检查点管理器
        :params: model_path_format 模型的存储路径格式
        :params: checkpoints_path_format 训练检查点的存储路径格式
        :params: save_interval 存储回合数间隔
        :return: CheckpointManager
        '''
        self.model_path_format = model_path_format
        self.checkpoints_path_format = checkpoints_path_format
        self.save_interval = save_interval

        self.model_dir = os.path.dirname(model_path_format)
        self.checkpoints_dir = os.path.dirname(checkpoints_path_format)
        create_dirs(self.model_dir)
        create_dirs(self.checkpoints_dir)


    def save_checkpoint(self, model, optimizer, epoch, epsilon, info={}, force=False) :
        '''
        保存训练检查点。
        但是若未满足训练回合数，不会进行保存。
        :params: model 正在训练的神经网络模型
        :params: optimizer 用于训练神经网络的优化器
        :params: epoch 已训练回合数
        :params: epsilon 当前探索率
        :params: info 其他附加参数
        :params: force 强制保存
        :return: 是否保存了检查点
        '''
        is_save = False
        if force or (epoch > 0 and epoch % self.save_interval == 0) :
            log.info(f"已训练 [{epoch}] 回合: ")

            checkpoint_path = self.checkpoints_path_format % epoch
            torch.save({
                KEY_MODEL_STATE_DICT: model.state_dict(),
                KEY_OPTIMIZER_STATE_DICT: optimizer.state_dict(),
                KEY_EPOCH: epoch,
                KEY_EPSILON: epsilon,
                KEY_INFO: info,
            }, checkpoint_path)
            is_save = True
            log.info(f"  自动存储检查点: {checkpoint_path}")

            model_path = self.model_path_format % epoch
            torch.save(model.state_dict(), model_path)
            log.info(f"  自动存储模型: {model_path}")
        return is_save


    def load_last_checkpoint(self) -> Checkpoint :
        '''
        加载最后一次记录的训练检查点
        :return: 检查点对象
        '''
        last_time = ''
        checkpoint_names = [f for f in os.listdir(self.checkpoints_dir) if f.endswith(CHECKPOINT_SUFFIX)]
        for cpn in checkpoint_names :
            _datetime = re.search(r'\d+', cpn)[0]
            last_time = max(last_time, _datetime)

        checkpoint = None
        if last_time :
            checkpoint_path = ''
            for cpn in checkpoint_names :
                if last_time in cpn :
                    checkpoint_path = os.path.join(self.checkpoints_dir, cpn)
                    break
            checkpoint = self.load_checkpoint(checkpoint_path)
        return checkpoint
    

    def load_checkpoint(self, checkpoint_path) -> Checkpoint :
        '''
        加载训练检查点
        :params: checkpoint_path 检查点的记录路径
        :return: 检查点对象
        '''
        checkpoint = None
        if os.path.exists(checkpoint_path) :
            cp = torch.load(checkpoint_path)
            checkpoint = Checkpoint(
                model_state_dict = cp.get(KEY_MODEL_STATE_DICT), 
                optimizer_state_dict = cp.get(KEY_OPTIMIZER_STATE_DICT), 
                epoch = cp.get(KEY_EPOCH), 
                epsilon = cp.get(KEY_EPSILON), 
                info = cp.get(KEY_INFO, {})
            )
            log.warn(f"已加载检查点：{checkpoint_path}")
        return checkpoint
        

    def _checkpoint_name(self, epoch=0) :
        '''
        构造检查点名称
        :params: epoch 已训练回合数
        :return: 检查点名称
        '''
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        return f'{CHECKPOINT_PREFIX}_{now}_{epoch}{CHECKPOINT_SUFFIX}'

