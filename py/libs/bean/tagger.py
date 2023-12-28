#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------


import cv2
import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
from conf.settings import *
from tools.utils import *


class Tagger :

    def __init__(self, env_name='Gym', eval=False) -> None :
        '''
        初始化标签记录器
        :params: env_name 环境名称，用于窗口显示名字
        :params: eval true: 评估模式； False: 训练模式
        :return: Tagger
        '''
        self.win_title = f"{env_name} ['Q' for exit]"
        self.eval = eval
        self.exit_button = ord('Q')
        self.fps = 60
        self.frames = []


    def reset(self) :
        '''
        清空已记录的帧
        :return: None
        '''
        self.frames.clear()


    def show(self, frame, labels=[]) :
        '''
        在指定帧的图像中添加信息标签、并渲染显示
        :params: frame 通过 env.render() 得到的帧，需要为 rgb_array 模式
        :params: labels 期望添加的标签
        :return: PIL 图像
        '''
        pil_img = self._add_labels(frame, labels)
        cv2_img = self._pil_to_cv2(pil_img)

        self.frames.append(pil_img)
        is_exit = False
        cv2.imshow(self.win_title, cv2_img)
        if cv2.waitKey(1) & 0xFF == self.exit_button:
            is_exit = True
        return is_exit


    def _add_labels(self, frame, labels=[]) :
        '''
        在指定帧的图像中添加信息标签
        :params: frame 通过 env.render() 得到的帧，需要为 rgb_array 模式
        :params: labels 期望添加的标签
        :return: PIL 图像
        '''
        # 把 gym 环境返回的渲染帧，转换为 PIL 图像
        image = Image.fromarray(frame)
        font_color = self._get_font_color(image)
        pos = self._get_font_pos(image)
        info = "\n".join(labels)

        # 在图像上描绘模型的训练/测试信息
        drawer = ImageDraw.Draw(image)
        drawer.text(pos, info, fill=font_color)
        return image
    

    def _get_font_color(self, image: Image) :
        '''
        根据图像整体颜色的明暗情况，返回人类可见的文字 RGB 颜色
        :params: image PIL 图像
        :return: RGB 颜色
        '''
        # 默认文字颜色为白色
        font_color = (0, 0, 0)

        # 如果图像的平均像素值偏暗，则文字颜色为白色
        if np.mean(image) < 128 :
            font_color = (255, 255, 255)
        return font_color
    

    def _get_font_pos(self, image: Image) :
        '''
        返回图片左上角（距离 left 1/20、 top 1/18）的位置坐标作为文字信息位置
        :params: image PIL 图像
        :return: pos 文字信息位置
        '''
        # 把信息放在图像左上角（距离 left 1/20、 top 1/18 的位置）
        pos = (image.size[0]/20, image.size[1]/18) 
        return pos
    

    def _pil_to_cv2(self, pil_img):
        '''
        PIL 图像转 cv2 图像
        :params: pil_img PIL 图像，PIL.Image
        :return: cv2 图像，numpy.ndarray
        '''
        cv2_img = np.array(pil_img)
        return cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)



    def save_ui(self, epoch) :
        '''
        保存智能体第 epoch 回合渲染的动作 UI 到 GIF
        :params: epoch 回合数
        :return: None
        '''
        path = get_render_ui_path(MODEL_NAME, epoch, self.eval)
        dir = os.path.dirname(path)
        create_dirs(dir)
        imageio.mimwrite(path, self.frames, duration=self.fps)
