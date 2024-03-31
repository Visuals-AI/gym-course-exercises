#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------


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
        # if not self._reinit(epoch) :
        #     return obs
        
        # --------------------------
        # 通过循环控制获取期望的初始状态
        # --------------------------
        # while True :
        #     x = obs[0][0]
        #     y = obs[0][1]
        #     v = obs[0][2]
        #     y = 1 if y >= 0 else -1
        #     v = 1 if v >= 0 else -1

        #     if x > 0 and y * v < 0 :
        #         break

        #     # 重新生成初始状态
        #     obs = self.env.reset()
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
    
