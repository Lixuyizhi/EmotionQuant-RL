#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测集成模块
负责将训练好的模型接入回测系统
"""

import argparse
import os
import re
import yaml
import pandas as pd
import logging

from src.models.rl_trainer import RLTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_model_filename(model_filename):
    # 支持两种格式：
    # 1. A2C_MaxWeightTradingEnv_model.zip
    # 2. best/A2C_MaxWeightTradingEnv_best_model/best_model.zip
    # 先尝试第一种
    match = re.match(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)_model\.zip", model_filename)
    if match:
        algo, env_name = match.group(1), match.group(2)
        return algo, env_name
    # 再尝试第二种
    match2 = re.search(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)_best_model", model_filename)
    if match2:
        algo, env_name = match2.group(1), match2.group(2)
        return algo, env_name
    raise ValueError("模型文件名格式不正确，应为: 算法_环境_model.zip 或 .../算法_环境_best_model/best_model.zip")

def main():
    parser = argparse.ArgumentParser(description="RL模型回测集成")
    parser.add_argument('--data', type=str, default=None, help="回测数据文件路径（如 data/SC_2020-01-01_2024-01-01.csv）")
    parser.add_argument('--model', type=str, default=None, help="已训练模型文件路径（如 models/A2C_MaxWeightTradingEnv_model.zip）")
    parser.add_argument('--config', type=str, default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 1. 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 获取数据和模型路径，优先命令行参数，否则用config.yaml
    data_path = args.data if args.data else config['backtest'].get('data_path')
    model_path = args.model if args.model else config['backtest'].get('model_path')
    if not data_path or not model_path:
        raise ValueError("请在命令行或config.yaml中指定回测数据文件路径和模型路径！")

    # 3. 直接加载已处理好的回测数据
    df = pd.read_csv(data_path)

    # 4. 解析模型文件名，确定算法和环境
    algo, env_name = parse_model_filename(os.path.basename(model_path))
    env_module = "signal_weight_env" if "SignalWeight" in env_name else "max_weight_env"
    config['model_training']['model']['algorithm'] = algo
    config['model_training']['model']['env_module'] = env_module
    config['model_training']['model']['env_name'] = env_name

    # 5. 创建环境
    trainer = RLTrainer(args.config)
    env = trainer.create_env(df, config=config)

    # 6. 加载模型
    model = trainer.load_model(model_path, algo)

    # 7. 回测循环
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    done = False
    total_reward = 0
    step_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, done, truncated, info = step_result
            done = done or truncated
        total_reward += reward
        step_count += 1
    stats = env.get_portfolio_stats()
    print("回测结果：", stats)

if __name__ == "__main__":
    main() 