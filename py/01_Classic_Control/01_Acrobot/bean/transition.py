#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------


from collections import namedtuple

# 创建一个具名元组，用于表示每个经验样本
Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'next_obs', 'done'))
