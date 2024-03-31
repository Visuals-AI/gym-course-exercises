# Pendulum

> https://gymnasium.farama.org/environments/classic_control/pendulum/

------

## 说明

Pendulum 是一个倒立摆问题：该系统由一个摆锤组成，摆锤的一端连接到固定点，另一端绕着固定点自由摆动。

摆锤从一个随机位置开始，目标是在自由端施加扭矩，使其摆动到重心位于固定点的正上方的垂直位置，然后坚持得越久越好。


## 模型训练

- 训练命令: `python py/01_Classic_Control/05_Pendulum/train_TD3.py -d 0.999`
- 查看训练过程参数: 
    - `tensorboard --logdir=./out/tensor/pendulum`
    - http://localhost:6006/
- 模型输出目录: `./out/models/pendulum` （默认每 500 回合保存一次）


## 模型测试

- 测试所有模型: `python py/01_Classic_Control/05_Pendulum/test_TD3.py`
- 测试单个模型: `python py/01_Classic_Control/05_Pendulum/test_TD3.py -m ${model_epoch}`
