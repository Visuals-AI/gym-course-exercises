#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------


from conf.settings import MAX_STEP


# 判断摆锤是否触发终止状态
class TerminateDetector :

    # 理论上完成一轮游戏，摆锤最多只会经过 3 个象限（允许在一二象限微小摆动），最少 1 个象限
    # 若摆锤经过全部 4 个象限，说明摆锤旋转了一圈；即使没旋转，也浪费步数，不是最优解
    # 但事实上摆锤在最低点来回借力时，确实需要经过四个象限，因此不能依赖此判断
    MAX_QUADRANT = 4

    # 若浪费步数在 三四 象限徘徊，明显不是最优解
    MAX_WANDER = MAX_STEP / 4

    # 限制最大角速度
    MAX_V = 6

    # 若超过一定步数都是最大速度，说明摆锤在转圈
    MAX_SPEED = MAX_STEP / 4

    def __init__(self) :
        self.quadrants = [False] * 4    # 摆锤经过的象限
        self.cnt34 = 0                  # 三四象限徘徊计数
        self.cnt_maxv = 0               # 最大角速度计数


    # 更新摆锤状态
    def update(self, x, y, v, a):
        if x < 0 :
            self.cnt34 += 1

        if abs(v) >= self.MAX_V :
            self.cnt_maxv += 1

        if self.in_quadrant1(x, y) :
            self.quadrants[0] = True
        elif self.in_quadrant2(x, y) :
            self.quadrants[1] = True
        elif self.in_quadrant3(x, y) :
            self.quadrants[2] = True
        elif self.in_quadrant4(x, y) :
            self.quadrants[3] = True


    def in_quadrant1(self, x, y) :
        return x >= 0 and y < 0
    
    def in_quadrant2(self, x, y) :
        return x > 0 and y >= 0

    def in_quadrant3(self, x, y) :
        return x <= 0 and y > 0
    
    def in_quadrant4(self, x, y) :
        return x < 0 and y <= 0


    def is_wander(self) :
        return self.cnt34 > self.MAX_WANDER


    def is_overspeed(self) :
        return self.cnt_maxv > self.MAX_SPEED
    

    def is_rotation(self) :
        return sum(self.quadrants) > self.MAX_QUADRANT


    def is_terminate(self) :
        # return self.is_wander() or self.is_rotation()
        return self.is_wander() or self.is_overspeed()

    