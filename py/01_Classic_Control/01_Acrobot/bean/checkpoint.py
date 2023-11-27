#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------


import os
import torch

class CheckpointManager:
    def __init__(self, model, optimizer, epsilon, save_dir='checkpoints', save_interval=10):
        self.model = model
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.save_dir = save_dir
        self.save_interval = save_interval

        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save_checkpoint(self, episode):
        if (episode + 1) % self.save_interval == 0:
            checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{episode}.pth')
            torch.save({
                'episode': episode,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                # 其他需要保存的参数
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            # 加载其他状态（如果有的话）
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

