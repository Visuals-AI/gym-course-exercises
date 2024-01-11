# Mountain Car（离散动作）

> https://gymnasium.farama.org/environments/classic_control/mountain_car/

------

## 说明

Mountain Car 是一种确定性 MDP（马尔可夫决策过程）问题。

目标是控制一个无法直接攀登陡峭山坡的小车，使其到达山顶。

但是小车的动力不足以直接爬上山坡，所以必须利用山坡的反向坡度来获得足够的动量。


## 模型训练

- 训练命令: `python py/01_Classic_Control/04_Mountain_Car/train_DQN.py`
- 查看训练过程参数: 
    - `tensorboard --logdir=./out/tensor/cart_pole`
    - http://localhost:6006/
- 模型输出目录: `./out/models/cart_pole` （默认每 500 回合保存一次）

> 默认使用 DQN 算法解题


## 模型测试

- 测试所有模型: `python py/01_Classic_Control/04_Mountain_Car/test_DQN.py`
- 测试单个模型: `python py/01_Classic_Control/04_Mountain_Car/test_DQN.py -m ./out/models/cart_pole/cart_pole_model_epoch_xxxx.pth`
