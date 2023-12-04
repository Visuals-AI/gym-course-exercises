# Gym-course-exercises
> OpenAI Gym 课程练习笔记

------


## 教程

https://gymnasium.farama.org/
https://github.com/dennybritz/reinforcement-learning


## 环境搭建

brew install swig

> windows 需要把 swig.exe 所在目录加入环境变量

conda create -n RL_GYM python=3.11.5
conda activate RL_GYM
pip install -r requirements.txt


tensorboard --logdir=runs



正向传播：定义了如何从输入数据获得输出的过程
反向传播：指计算损失函数相对于网络参数的梯度的过程。
损失
梯度
梯度裁剪
梯度下降
梯度爆炸

