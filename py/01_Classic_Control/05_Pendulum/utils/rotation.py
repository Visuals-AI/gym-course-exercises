#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------


# 判断摆锤是否旋转
class RotationDetector :

    # 本题在扭矩最大的情况下，大概 20 step 一圈
    # 但事实上根本不需要绕过 3/4 圈就能从顺时针或逆时针移动到顶端
    ROTATION_LIMIT = 10

    # 最低的旋转速度
    VELOCITY_LIMIT = 1

    def __init__(self) :
        self.clockwise = 0      # 顺时针计数器
        self.re_clockwise = 0   # 逆时针计数器


    # 更新摆锤状态
    def update(self, x, y, v, a):

        if abs(v) >= self.VELOCITY_LIMIT :

            # 逆时针旋转
            if v > 0 and a > 0 :
                self.re_clockwise += 1
                self.clockwise = 0
                # print(f'逆时针旋转: {self.re_clockwise}, x: {x}, a: {a}, v: {v}')

            # 顺时针旋转
            elif v < 0 and a < 0 :
                self.clockwise += 1
                self.re_clockwise = 0
                # print(f'顺时针旋转: {self.clockwise}, x: {x}, a: {a}, v: {v}')

            else :
                self.clockwise = 0
                self.re_clockwise = 0
        
        is_rotation = (self.clockwise > self.ROTATION_LIMIT or self.re_clockwise > self.ROTATION_LIMIT)
        if is_rotation :
            self.clockwise = 0
            self.re_clockwise = 0
            # print(f"旋转了一圈")
        return is_rotation
