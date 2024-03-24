#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2024/03/23 13:52
# -----------------------------------------------


# 判断 旋转
# v 大于阈值
# 如果逆时针旋转，x > 0 时扭矩为 -， x < 0 时扭矩为 +
# 如果顺时针旋转，x > 0 时扭矩为 +， x < 0 时扭矩为 -
class RotationDetector :

    # 本题在扭矩最大的情况下，大概 20 step 一圈
    ROTATION_LIMIT = 20

    # 最低的旋转速度
    VELOCITY_LIMIT = 1

    def __init__(self) :
        self.clockwise = 0      # 顺时针计数器
        self.re_clockwise = 0   # 逆时针计数器


    def update(self, obs, action):
        x = obs[0][0]   # 自由端的 x 坐标
        y = obs[0][1]   # 自由端的 y 坐标
        v = obs[0][2]   # 自由端的角速度，顺时针为- 逆时针为+ 
        a = action[0]   # 对自由端施加的扭矩

        if v >= self.VELOCITY_LIMIT :

            # 逆时针旋转
            if (x > 0 and a < 0) or (x < 0 and a > 0) :
                self.re_clockwise += 1
                self.clockwise = 0
                # print(f'逆时针旋转: {self.re_clockwise}, x: {x}, a: {a}, v: {v}')

            # 顺时针旋转
            elif (x > 0 and a > 0) or (x < 0 and a < 0) :
                self.clockwise += 1
                self.re_clockwise = 0
                # print(f'顺时针旋转: {self.clockwise}, x: {x}, a: {a}, v: {v}')

            else :
                self.clockwise = 0
                self.re_clockwise = 0
        
        return (self.clockwise >= self.ROTATION_LIMIT or self.re_clockwise >= self.ROTATION_LIMIT)
