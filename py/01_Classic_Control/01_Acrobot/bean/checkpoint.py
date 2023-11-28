#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------


import re
import os
import datetime
import torch
from conf.settings import *
from tools.utils import *
from color_log.clog import log


KEY_MODEL_STATE_DICT = 'model_state_dict'
KEY_OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
KEY_EPISODE = 'episode'
KEY_EPSILON = 'epsilon'
KEY_INFO = 'info'


class Checkpoint :

    def __init__(self, model_state_dict, optimizer_state_dict, episode, epsilon, info={}) -> None:
        self.model_state_dict = model_state_dict
        self.optimizer_state_dict = optimizer_state_dict
        self.episode = episode      # 已训练的回合数（迭代数）
        self.epsilon = epsilon      # 当前探索率
        self.info = info            # 其他信息
    


class CheckpointManager:

    def __init__(self, checkpoints_dir=CHECKPOINTS_DIR, 
                 save_interval=SAVE_CHECKPOINT_INTERVAL) :
        self.checkpoints_dir = checkpoints_dir
        self.save_interval = save_interval
        create_dirs(checkpoints_dir)


    def save_checkpoint(self, model, optimizer, episode, epsilon, info={}) :
        if episode % self.save_interval == 0 :
            checkpoint_path = os.path.join(self.checkpoints_dir, self._checkpoint_name())
            torch.save({
                KEY_MODEL_STATE_DICT: model.state_dict(),
                KEY_OPTIMIZER_STATE_DICT: optimizer.state_dict(),
                KEY_EPISODE: episode,
                KEY_EPSILON: epsilon,
                KEY_INFO: info,
            }, checkpoint_path)
            log.info(f"已训练 [{episode}] 回合，自动存储检查点: {checkpoint_path}")


    def load_last_checkpoint(self) -> Checkpoint :
        last_idx = 0
        checkpoint_names = [f for f in os.listdir(self.checkpoints_dir) if f.endswith(CHECKPOINT_SUFFIX)]
        for cpn in checkpoint_names :
            idx = int(re.search(r'\d+', cpn)[0])
            last_idx = max(last_idx, idx)

        checkpoint = None
        if last_idx > 0 :
            checkpoint_path = os.path.join(self.checkpoints_dir, self._checkpoint_name(last_idx))
            checkpoint = self.load_checkpoint(checkpoint_path)
        return checkpoint
    

    def load_checkpoint(self, checkpoint_path) -> Checkpoint :
        checkpoint = None
        if os.path.exists(checkpoint_path) :
            cp = torch.load(checkpoint_path)
            checkpoint = Checkpoint(
                model_state_dict = cp.get(KEY_MODEL_STATE_DICT), 
                optimizer_state_dict = cp.get(KEY_OPTIMIZER_STATE_DICT), 
                episode = cp.get(KEY_EPISODE), 
                epsilon = cp.get(KEY_EPSILON), 
                info = cp.get(KEY_INFO, {})
            )
            log.warn(f"已加载上次训练的检查点：{checkpoint_path}")
        return checkpoint
        

    def _checkpoint_name(self, idx='') :
        if not idx :
            idx = datetime.now().strftime("%Y%m%d%H%M%S")
        return f'{CHECKPOINT_PREFIX}_{idx}{CHECKPOINT_SUFFIX}'



