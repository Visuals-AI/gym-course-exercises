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


import argparse
import torch
import gymnasium as gym
from bean.train_args import TrainArgs
from tools.utils import *
from conf.settings import *
from color_log.clog import log


def arguments() :
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='Gym - Acrobot 测试脚本',
        description='在默认环境下、使用深度 Q 网络（DQN）验证智能体操作 Acrobot', 
        epilog='\r\n'.join([
            '运行环境: python3', 
            '运行示例: python py/01_Classic_Control/01_Acrobot/test_DQN.py'
        ])
    )
    parser.add_argument('-r', '--render', dest='render', action='store_false', default=True, help='渲染模式: 可以通过 GUI 观察智能体实时交互情况，但是会极大拉低训练效率')
    parser.add_argument('-c', '--cpu', dest='cpu', action='store_true', default=False, help='强制使用 CPU: 默认情况下，自动优先使用 GPU 训练（除非没有 GPU）')
    return parser.parse_args()


def main(args) :
    env = gym.make('Acrobot-v1', 
                    render_mode=("human" if args.render else None)
    )

    run_ai(args, env)
    

def run_ai(args, env) :
    targs = TrainArgs(args, env, 
                      eval=True     # 设置为评估模式
    )
    
    targs.model.load_state_dict(         # 加载模型参数  
                torch.load(ACROBOT_MODEL_PATH)
            )
    
    ACROBOT_V1_MAX_STEP = 200
    step_counter = 0

    obs = env.reset()
    for _ in range(ACROBOT_V1_MAX_STEP) :

        # 渲染 GUI（前提是 env 初始化时使用 human 模式）
        env.render()
        
        # 将当前状态转换为适当的输入格式
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs = torch.from_numpy(obs).float().unsqueeze(0)

        # state = to_tensor(raw_obs[0], targs)  # 把观测空间状态数组送入神经网络所在的设备
        
        obs = obs.to(targs.device)

        # 使用模型预测动作
        with torch.no_grad():
            action = targs.model(obs).max(1)[1].view(1, 1).item()
        
        # 执行动作并获取下一个状态
        next_obs, _, done, _, _ = env.step(action)
        step_counter +=1
        print(step_counter)
        if done:
            break
        obs = next_obs

    env.close()
    

if __name__ == '__main__' :
    main(arguments())
