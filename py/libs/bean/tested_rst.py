#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------



class TestedResult :

    def __init__(self, epoch, min_step, max_step, avg_step, percentage) -> None:
        self.epoch = epoch
        self.min_step = min_step
        self.max_step = max_step
        self.avg_step = avg_step
        self.percentage = percentage


    def __repr__(self) -> str:
        return f"  模型 [{self.epoch}] 测试结果: [成功率={self.percentage:.2f}%][最小步数={self.min_step}][最大步数={self.max_step}][平均步数={self.avg_step}]"
