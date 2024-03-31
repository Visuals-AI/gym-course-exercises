# Bipedal Walker

> https://gymnasium.farama.org/environments/box2d/bipedal_walker/

------

## 说明

Bipedal Walker 双足步行者

## 模型训练

- 训练命令: `python py/01_box2d/01_Bipedal_Walker/train_TD3.py -d 0.999`
- 查看训练过程参数: 
    - `tensorboard --logdir=./out/tensor/bipedal_walker`
    - http://localhost:6006/
- 模型输出目录: `./out/models/bipedal_walker` （默认每 500 回合保存一次）


## 模型测试

- 测试所有模型: `python py/01_box2d/01_Bipedal_Walker/test_TD3.py`
- 测试单个模型: `python py/01_box2d/01_Bipedal_Walker/test_TD3.py -m ${model_epoch}`
