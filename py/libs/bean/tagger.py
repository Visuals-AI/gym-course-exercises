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


class Tagger :

    def __init__(self) -> None :
        self.win_title = 'Gym'
        self.exit_button = ord('q')
        self.frames = []


    def show(self, frame, labels=[]) :
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
        初始化检查点管理器
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
        # 默认文字颜色为白色
        font_color = (0, 0, 0)

        # 如果图像的平均像素值偏暗，则文字颜色为白色
        if np.mean(image) < 128 :
            font_color = (255, 255, 255)
        return font_color
    

    def _get_font_pos(self, image: Image) :
        # 把信息放在图像左上角（距离 left 1/20、 top 1/18 的位置）
        pos = (image.size[0]/20, image.size[1]/18) 
        return pos
    

    def _pil_to_cv2(self, pil_img):
        '''
        PIL 图像转 cv2 图像
        [param] pil_img: PIL图像，PIL.Image
        :return: cv2 图像，numpy.ndarray
        '''
        cv2_img = np.array(pil_img)
        return cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)



    def seve_gif(self, epoch) :
        path = os.path.join('./out/videos/', f'random_agent_{epoch}.gif')
        fps = 60
        imageio.mimwrite(path, self.frames, duration=fps)
