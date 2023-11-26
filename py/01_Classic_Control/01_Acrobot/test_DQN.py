#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------
# 经典控制： Acrobot （杂技机器人）
#   Acrobot 是一个双节摆问题，目标是用最少的步骤使得摆的末端达到一定高度。
# 
# 相关文档：
#   https://gymnasium.farama.org/environments/classic_control/acrobot/
#   http://incompleteideas.net/book/11/node4.html
# -----------------------------------------------


import torch
import gymnasium as gym
from bean.dqn import DQN
from conf.settings import *
from color_log.clog import log



def main() :
    env = gym.make('Acrobot-v1', render_mode="human")   # human 表示启动交互界面
    

def run_ai(env) :
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 创建模型实例
    model = DQN(state_size, action_size)

    # 加载模型参数
    model.load_state_dict(torch.load(TRIAN_MODEL_PATH))

    # 设置为评估模式
    model.eval()

    # 运行智能体
    state = env.reset()
    for _ in range(1000):  # 设置一个足够长的时间步骤，或者直到环境结束
        env.render()  # 渲染 GUI，前提是 env 初始化时使用 human 模式
        
        # 将当前状态转换为适当的输入格式
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # 使用模型预测动作
        with torch.no_grad():
            action = model(state).max(1)[1].view(1, 1).item()
        
        # 执行动作并获取下一个状态
        state, _, done, _, _ = env.step(action)
        
        if done:
            state = env.reset()

    env.close()
    

if __name__ == '__main__' :
    main()
