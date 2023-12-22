# Mountain Car

> https://gymnasium.farama.org/environments/classic_control/mountain_car/

------

## 说明

FIXME

## 模型训练

- 训练命令: `python py/01_Classic_Control/04_Mountain_Car/train_DQN.py`
- 查看训练过程参数: 
    - `tensorboard --logdir=./out/tensor/mountain_car`
    - http://localhost:6006/
- 模型输出目录: `./out/models/mountain_car` （默认每 500 回合保存一次）

> 默认使用 DQN 算法解题


## 模型测试

- 测试所有模型: `python py/01_Classic_Control/04_Mountain_Car/test_DQN.py`
- 测试单个模型: `python py/01_Classic_Control/04_Mountain_Car/test_DQN.py -m ./out/models/mountain_car/mountain_car_model_epoch_xxxx.pth`
