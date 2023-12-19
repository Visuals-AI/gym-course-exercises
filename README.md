# Gym-course-exercises
> OpenAI Gym 课程练习笔记

------


## 0x00 课程资源

- docs: https://gymnasium.farama.org/
- github: https://github.com/dennybritz/reinforcement-learning


## 0x10 环境搭建

> 详细可参考配套教程《[「Gym 课程笔记 01」环境搭建与基本概念](https://exp-blog.com/ai/gym-bi-ji-01-huan-jing-da-jian-yu-ji-ben-gai-nian/)》

### 0x11 安装 conda（管理 python 虚拟环境）

1. 安装 [miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)
2. 把 conda 安装目录下的 Scripts 目录添加到 Path 环境变量
3. 初始化终端 `conda init powershell` （powershell 应更改为你当前使用的终端类型）
4. 查看 CUDA 版本: `nvidia-smi`
5. 到 [pytorch](https://pytorch.org/get-started/locally/) 官网，生成 RL 环境的安装命令
6. 执行生成的命令、在 conda 中安装 pytroch, 如命令为: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
7. 安装 [swig](https://open-box.readthedocs.io/zh-cn/latest/installation/install_swig.html)
8. 把 swig 的安装目录添加到 Path 环境变量

### 0x12 创建 Gym 虚拟环境


- 创建名为 RL_GYM 的 python3 虚拟环境: `conda create -n RL_GYM python=3.11.5`
- 激活并切换到 RL_GYM 环境: `conda activate RL_GYM` （每次退出终端后需要重新激活）
- 安装 Gym 依赖: `pip install -r requirements.txt`




## 0x20 训练

`python py/01_Classic_Control/01_Acrobot/train_DQN.py`

查看训练模型的过程参数: `tensorboard --logdir=runs`



