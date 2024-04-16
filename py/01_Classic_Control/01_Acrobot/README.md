# Acrobot

> https://gymnasium.farama.org/environments/classic_control/acrobot/

------

## 说明

Acrobot 是一个双节摆问题，目标是用最少的步骤使得摆的末端达到一定高度。


## 模型训练

- 训练命令: `python py/01_Classic_Control/01_Acrobot/train_DQN.py`
- 查看训练过程参数: 
    - `tensorboard --logdir=./out/tensor/acrobot`
    - http://localhost:6006/
- 模型输出目录: `./out/models/acrobot` （默认每 500 回合保存一次）

> 默认使用 DQN 算法解题


## 模型测试

- 测试所有模型: `python py/01_Classic_Control/01_Acrobot/test_DQN.py`
- 测试单个模型: `python py/01_Classic_Control/01_Acrobot/test_DQN.py -m ./out/models/acrobot/acrobot_model_epoch_xxxx.pth`


> 验证训练好的最优模型: `python py/01_Classic_Control/01_Acrobot/test_DQN.py -m ./optimal/01_Classic_Control/01_Acrobot/models/acrobot_model_epoch_2000.pth -a`

