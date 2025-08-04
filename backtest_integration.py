#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测集成模块
负责将训练好的模型接入回测系统，使用独立数据集进行回测
"""

import argparse
import os
import re
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from src.models.rl_trainer import RLTrainer
from src.data.data_loader import OilDataLoader
from src.features.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_model_filename(model_filename):
    """解析模型文件名，支持两种格式：
    1. PPO_SignalWeightTradingEnv_model.zip
    2. best_model.zip (在best目录下)
    """
    logger.info(f"解析模型文件名: {model_filename}")
    
    # 格式1: 直接模型文件 (如: PPO_SignalWeightTradingEnv_model.zip)
    match1 = re.match(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)_model\.zip", model_filename)
    if match1:
        algo, env_name = match1.group(1), match1.group(2)
        logger.info(f"匹配格式1: 算法={algo}, 环境={env_name}")
        return algo, env_name
    
    # 格式2: best_model.zip (在best目录下)
    if model_filename == "best_model.zip":
        # 从完整路径中提取信息
        # 完整路径格式: models/best/PPO_SignalWeightTradingEnv_best_model/best_model.zip
        # 需要从路径中提取算法和环境信息
        logger.warning("检测到best_model.zip格式，需要从路径中提取算法和环境信息")
        # 这里需要特殊处理，暂时返回默认值
        return "PPO", "SignalWeightTradingEnv"
    
    # 格式3: 从完整路径中提取 (如: models/best/PPO_SignalWeightTradingEnv_best_model/best_model.zip)
    # 提取路径中的算法和环境信息
    path_match = re.search(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)_best_model", model_filename)
    if path_match:
        algo, env_name = path_match.group(1), path_match.group(2)
        logger.info(f"从路径中提取: 算法={algo}, 环境={env_name}")
        return algo, env_name
    
    # 如果都不匹配，尝试更宽松的匹配
    match3 = re.search(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)", model_filename)
    if match3:
        algo, env_name = match3.group(1), match3.group(2)
        logger.warning(f"使用宽松匹配解析模型文件名: {model_filename}")
        return algo, env_name
    
    raise ValueError(f"模型文件名格式不正确: {model_filename}，应为: 算法_环境_model.zip 或 .../算法_环境_best_model/best_model.zip")

def prepare_backtest_data(config, start_date=None, end_date=None, data_source='local'):
    """准备回测数据
    
    Args:
        config: 配置字典
        start_date: 回测开始日期 (格式: 'YYYY-MM-DD')
        end_date: 回测结束日期 (格式: 'YYYY-MM-DD')
        data_source: 数据源类型 ('local' 或 'akshare')
        
    Returns:
        df_backtest: 回测数据DataFrame
    """
    logger.info("=" * 50)
    logger.info("准备回测数据...")
    logger.info("=" * 50)
    
    try:
        # 1. 加载原始数据
        loader = OilDataLoader()
        
        # 如果指定了日期范围，临时修改配置
        if start_date and end_date:
            original_start = loader.start_date
            original_end = loader.end_date
            loader.start_date = start_date
            loader.end_date = end_date
            logger.info(f"使用指定时间范围: {start_date} 到 {end_date}")
        
        # 2. 获取数据
        df = loader.get_oil_futures_data()
        
        # 恢复原始配置
        if start_date and end_date:
            loader.start_date = original_start
            loader.end_date = original_end
        
        logger.info(f"回测数据加载完成，数据形状: {df.shape}")
        logger.info(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
        
        # 3. 特征工程
        logger.info("进行特征工程...")
        engineer = FeatureEngineer()
        
        # 添加技术指标
        df_features = engineer.add_technical_indicators(df)
        logger.info(f"技术指标添加完成，特征数量: {len(df_features.columns) - 6}")
        
        # 添加目标变量
        df_features = engineer.add_target_variables(df_features)
        logger.info("目标变量添加完成")
        
        # 创建交易特征
        df_features = engineer.create_trading_features(df_features)
        logger.info("交易特征创建完成")
        
        # 4. 准备回测数据
        logger.info("准备回测数据...")
        X, y = engineer.prepare_features_for_ml(df_features)
        
        # 将特征矩阵转换回DataFrame，保留日期信息
        df_backtest = df_features.copy()
        
        logger.info(f"回测数据准备完成，数据形状: {df_backtest.shape}")
        logger.info(f"特征数量: {X.shape[1]}")
        
        return df_backtest
        
    except Exception as e:
        logger.error(f"回测数据准备失败: {e}")
        raise

def run_backtest(model_path, df_backtest, config, save_results=True):
    """运行回测
    
    Args:
        model_path: 模型文件路径
        df_backtest: 回测数据
        config: 配置字典
        save_results: 是否保存回测结果
        
    Returns:
        stats: 回测统计结果
    """
    logger.info("=" * 50)
    logger.info("开始回测...")
    logger.info("=" * 50)
    
    try:
        # 1. 解析模型文件名，确定算法和环境
        # 如果文件名是best_model.zip，需要从完整路径中提取信息
        if os.path.basename(model_path) == "best_model.zip":
            # 从完整路径中提取算法和环境信息
            path_match = re.search(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)_best_model", model_path)
            if path_match:
                algo, env_name = path_match.group(1), path_match.group(2)
                logger.info(f"从完整路径中提取: 算法={algo}, 环境={env_name}")
            else:
                # 如果无法从路径提取，使用默认值
                algo, env_name = "PPO", "SignalWeightTradingEnv"
                logger.warning(f"无法从路径提取算法和环境信息，使用默认值: {algo}, {env_name}")
        else:
            # 使用文件名解析
            algo, env_name = parse_model_filename(os.path.basename(model_path))
        
        env_module = "signal_weight_env" if "SignalWeight" in env_name else "max_weight_env"
        
        logger.info(f"模型算法: {algo}")
        logger.info(f"环境名称: {env_name}")
        logger.info(f"环境模块: {env_module}")
        
        # 2. 更新配置
        config['model_training']['model']['algorithm'] = algo
        config['model_training']['model']['env_module'] = env_module
        config['model_training']['model']['env_name'] = env_name
        
        # 3. 确保回测使用与训练时相同的环境参数
        logger.info("确保回测环境参数与训练时一致...")
        
        # 获取训练时的环境参数
        training_env_params = config['model_training'].get(env_module, {})
        logger.info(f"训练时环境参数: {training_env_params}")
        
        # 如果回测配置中有不同的参数，使用训练时的参数
        if 'backtest' in config and 'env_params' in config['backtest']:
            backtest_env_params = config['backtest']['env_params'].get(env_module, {})
            if backtest_env_params:
                logger.warning(f"检测到回测参数与训练参数不同，将使用训练时的参数")
                logger.warning(f"回测参数: {backtest_env_params}")
                logger.info(f"使用训练参数: {training_env_params}")
        
        # 4. 创建环境（使用训练时的参数）
        trainer = RLTrainer("config/config.yaml")
        env = trainer.create_env(df_backtest, config=config)
        
        # 4. 加载模型
        model = trainer.load_model(model_path, algo)
        
        # 5. 回测循环
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        done = False
        total_reward = 0
        step_count = 0
        actions_taken = []
        rewards_history = []
        
        logger.info("开始回测循环...")
        
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
            actions_taken.append(action)
            rewards_history.append(reward)
            
            if step_count % 100 == 0:
                logger.info(f"回测进度: {step_count} 步, 累计奖励: {total_reward:.4f}")
        
        # 6. 获取回测统计
        stats = env.get_portfolio_stats()
        
        logger.info("=" * 50)
        logger.info("回测完成！")
        logger.info("=" * 50)
        logger.info(f"总步数: {step_count}")
        logger.info(f"总奖励: {total_reward:.4f}")
        logger.info(f"平均奖励: {total_reward/step_count:.4f}")
        
        # 7. 保存回测结果
        if save_results:
            save_backtest_results(stats, model_path, df_backtest, actions_taken, rewards_history)
        
        return stats
        
    except Exception as e:
        logger.error(f"回测失败: {e}")
        raise

def save_backtest_results(stats, model_path, df_backtest, actions_taken, rewards_history):
    """保存回测结果
    
    Args:
        stats: 回测统计结果
        model_path: 模型文件路径
        df_backtest: 回测数据
        actions_taken: 动作历史
        rewards_history: 奖励历史
    """
    try:
        # 创建结果目录
        results_dir = "backtest_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 生成结果文件名（包含算法和环境名）
        try:
            algo, env_name = parse_model_filename(model_path)
        except Exception:
            algo, env_name = "UnknownAlgo", "UnknownEnv"
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 如果是best_model，额外标注
        if model_name == "best_model":
            result_file = f"{results_dir}/{algo}_{env_name}_{model_name}_backtest_{timestamp}.json"
        else:
            result_file = f"{results_dir}/{algo}_{env_name}_final_backtest_{timestamp}.json"
        
        # 准备结果数据
        results = {
            "model_info": {
                "model_path": model_path,
                "model_name": model_name,
                "backtest_date": datetime.now().isoformat()
            },
            "data_info": {
                "data_shape": df_backtest.shape,
                "data_range": {
                    "start_date": df_backtest['date'].min().isoformat(),
                    "end_date": df_backtest['date'].max().isoformat()
                }
            },
            "backtest_stats": stats,
            "performance_metrics": {
                "total_steps": len(actions_taken),
                "total_reward": sum(rewards_history),
                "average_reward": np.mean(rewards_history),
                "reward_std": np.std(rewards_history),
                "max_reward": max(rewards_history),
                "min_reward": min(rewards_history)
            }
        }
        
        # 保存结果
        import json
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"回测结果已保存到: {result_file}")
        
    except Exception as e:
        logger.error(f"保存回测结果失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="RL模型回测集成")
    parser.add_argument('--model', type=str, default=None, help="已训练模型文件路径（如 models/PPO_SignalWeightTradingEnv_model.zip）")
    parser.add_argument('--config', type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument('--start_date', type=str, default=None, help="回测开始日期（格式: YYYY-MM-DD）")
    parser.add_argument('--end_date', type=str, default=None, help="回测结束日期（格式: YYYY-MM-DD）")
    parser.add_argument('--data_source', type=str, default='local', choices=['local', 'akshare'], help="数据源类型")
    parser.add_argument('--no_save', action='store_true', help="不保存回测结果")
    args = parser.parse_args()

    # 1. 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 获取模型路径
    model_path = args.model if args.model else config['backtest'].get('model_path')
    if not model_path:
        raise ValueError("请在命令行或config.yaml中指定模型路径！")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 3. 准备回测数据
    df_backtest = prepare_backtest_data(
        config, 
        start_date=args.start_date, 
        end_date=args.end_date, 
        data_source=args.data_source
    )

    # 4. 运行回测
    stats = run_backtest(
        model_path, 
        df_backtest, 
        config, 
        save_results=not args.no_save
    )

    # 5. 打印结果
    print("\n" + "=" * 60)
    print("回测结果摘要")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 