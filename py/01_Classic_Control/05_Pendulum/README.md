# Pendulum

> https://gymnasium.farama.org/environments/classic_control/pendulum/

------

## 说明

Pendulum 是一个倒立摆问题：该系统由一个摆锤组成，摆锤的一端连接到固定点，另一端绕着固定点自由摆动。

摆锤从一个随机位置开始，目标是在自由端施加扭矩，将其摆动到垂直位置，其重心位于固定点的正上方，然后坚持越久越好。

> 这题类似于 [Cart Pole](../02_Cart_Pole/) 问题，可以看作是 [Cart Pole](../02_Cart_Pole/) 的进阶版，且是连续动作空间问题


## 模型训练

- 训练命令: `python py/01_Classic_Control/05_Pendulum/train_TD3.py -d 0.999`
- 查看训练过程参数: 
    - `tensorboard --logdir=./out/tensor/pendulum`
    - http://localhost:6006/
- 模型输出目录: `./out/models/pendulum` （默认每 500 回合保存一次）

> 默认使用 TD3 算法解题，但是设定了更慢的探索衰减率 0.999；因为在本题中，摆锤越想接近目标、就必须先学会精细控制，更需要的是 “探索” 未知而非学习曾经失败的经验。


## 模型测试

- 测试所有模型: `python py/01_Classic_Control/05_Pendulum/test_TD3.py`
- 测试单个模型: `python py/01_Classic_Control/05_Pendulum/test_TD3.py -m ${model_epoch}`
