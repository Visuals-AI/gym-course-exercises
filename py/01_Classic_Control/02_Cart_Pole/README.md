# Cart Pole

> https://gymnasium.farama.org/environments/classic_control/cart_pole/

------

## 说明

Cart Pole 是一个倒立摆问题，目标是控制一个小车来平衡一个竖直安装在其上的杆子。

## 模型训练

- 训练命令: `python py/01_Classic_Control/02_Cart_Pole/train_DQN.py`
- 查看训练过程参数: 
    - `tensorboard --logdir=./out/tensor/cart_pole`
    - http://localhost:6006/
- 模型输出目录: `./out/models/cart_pole` （默认每 500 回合保存一次）

> 默认使用 DQN 算法解题


## 模型测试

- 测试所有模型: `python py/01_Classic_Control/02_Cart_Pole/test_DQN.py`
- 测试单个模型: `python py/01_Classic_Control/02_Cart_Pole/test_DQN.py -m ./out/models/cart_pole/cart_pole_model_epoch_xxxx.pth`
