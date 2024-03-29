#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------


# 中间难度起点奖励(SoID)
# 反映了强化学习（RL）中一个常见的问题：如何平衡正面奖励和负面奖励，以促进有效学习。
# 初始学习阶段如果主要是负面奖励，会导致学习速度慢，因为学习算法可能难以快速发现导致正面奖励的行为；而如果过多地依赖正面奖励，又可能导致算法忽略或不足够学习到避免不良行为的重要性。所描述的“中间难度起点奖励”（Start of Intermediate Difficulty，SoID）是一个旨在解决这个问题的策略，通过从一个“中间”难度的起点开始学习，以达到更平衡的学习进程。
# 具体来说，这种策略试图通过提供一个既不太难也不太易的起始点，使得学习算法可以更有效地学习到从初始状态到目标状态的转换，同时避免了一开始就遇到的过度挫败感或过于简单导致的学习效率低下。
# 在这题中，可以尝试通过设置一个中等难度的初始状态（例如，让摆锤从稍微偏离垂直向上的位置开始摆动），并适当地调整奖励机制，来实现 SoID 策略。
# 譬如要训练 10000 回合：
#   前 2000 回合设计 80% 从一二象限开始
#   随后 2000 回合 60% 从一二象限开始
#   再随后 2000 回合 40% 从一二象限开始
#   再随后 2000 回合 20% 从一二象限开始
#   最后 2000 回合完全随机
# 这样做可以在前期先建立智能体的行动策略框架，中后期逐步引入适当的难度增加挑战性，使得智能体学习效率和积极性更高

import random

class CtrlInitEnv :

    def __init__(self, env, max_epoches) -> None:
        self.env = env
        self.max_epoches = max_epoches
        self.stage_20 = int(max_epoches * 0.2)
        self.stage_40 = int(max_epoches * 0.4)
        self.stage_60 = int(max_epoches * 0.6)
        self.stage_80 = int(max_epoches * 0.8)


    # 因为摆锤受重力影响，总是趋向徘徊在三四象限，脱离三四象限的几率较低
    # 因此控制初始状态：
    #   前期尽量从一二象限开始，先建立智能体的行动策略框架
    #   中后期逐步引入三四象限的初始状态，使得智能体可以更好地挑战更高难度
    def reset(self, epoch) :
        obs = self.env.reset()
        if not self._reinit(epoch) :
            return obs
        
        # --------------------------
        # 通过循环控制获取期望的初始状态
        # --------------------------
        while True :
            x = obs[0][0]
            y = obs[0][1]
            v = obs[0][2]
            y = 1 if y >= 0 else -1
            v = 1 if v >= 0 else -1

            if x > 0 and y * v < 0 :
                break

            # 重新生成初始状态
            obs = self.env.reset()
        return obs


    def _reinit(self, epoch) :
        reinit = False

        # 在前 20% 的训练回合，控制 80% 的初始状态都是一二象限、速度向上
        if epoch < self.stage_20 :
            if self.p80() :
                reinit = True

        # 在前 40% 的训练回合，控制 60% 的初始状态都是一二象限、速度向上
        elif epoch < self.stage_40 :
            if self.p60() :
                reinit = True

        # 在前 60% 的训练回合，控制 40% 的初始状态都是一二象限、速度向上
        elif epoch < self.stage_60 :
            if self.p40() :
                reinit = True

        # 在前 80% 的训练回合，控制 20% 的初始状态都是一二象限、速度向上
        elif epoch < self.stage_80 :
            if self.p20() :
                reinit = True

        # 最后 20% 的训练回合，完全随机
        else :
            pass

        return reinit


    def p80(self) :
        return self.percentage(80)
    
    def p60(self) :
        return self.percentage(60)
    
    def p40(self) :
        return self.percentage(40)
    
    def p20(self) :
        return self.percentage(20)
    
    def percentage(self, ratio=100) :
        return random.randint(0, 100) < ratio
    
