#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测集成模块
负责将训练好的模型接入回测系统
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategies.backtrader_strategies import BacktraderBacktester
from src.models.rl_trainer import RLTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest_integration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("配置文件加载成功")
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        raise

def load_processed_data(data_dir: str = "data/processed") -> tuple:
    """加载已处理的数据"""
    logger.info(f"加载已处理的数据从: {data_dir}")
    
    try:
        df_features = pd.read_csv(f"{data_dir}/features_data.csv")
        X = pd.read_csv(f"{data_dir}/features_matrix.csv")
        y = pd.read_csv(f"{data_dir}/target_variable.csv", squeeze=True)
        
        logger.info(f"数据加载完成，特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
        
        return df_features, X, y
        
    except Exception as e:
        logger.error(f"加载已处理数据失败: {e}")
        raise

def load_trained_model(model_path: str = None, algorithm: str = None) -> object:
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        algorithm: 算法名称
        
    Returns:
        加载的模型对象
    """
    logger.info("加载训练好的模型...")
    
    try:
        # 如果没有指定模型路径，尝试从默认位置加载
        if model_path is None:
            # 尝试从训练结果中获取模型路径
            training_config_path = "models/training_results/training_config.json"
            if os.path.exists(training_config_path):
                import json
                with open(training_config_path, 'r', encoding='utf-8') as f:
                    training_config = json.load(f)
                model_path = training_config.get('model_path', '')
            
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError("未找到训练好的模型文件")
        
        # 如果没有指定算法，尝试从配置中获取
        if algorithm is None:
            config = load_config()
            algorithm = config['model']['algorithm']
        
        # 加载模型
        trainer = RLTrainer()
        model = trainer.load_model(model_path, algorithm)
        
        logger.info(f"模型加载成功: {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

def run_traditional_backtest(df_features: pd.DataFrame, config: dict) -> dict:
    """运行传统策略回测
    
    Args:
        df_features: 包含特征的DataFrame
        config: 配置字典
        
    Returns:
        回测结果字典
    """
    logger.info("=" * 50)
    logger.info("运行传统策略回测...")
    logger.info("=" * 50)
    
    try:
        # 创建回测器
        backtester = BacktraderBacktester()
        
        # 运行策略比较
        results = backtester.compare_strategies(df_features)
        
        # 保存回测结果
        results_dir = "results/traditional_backtest"
        os.makedirs(results_dir, exist_ok=True)
        
        import json
        with open(f"{results_dir}/backtest_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"传统回测结果已保存到: {results_dir}/backtest_results.json")
        
        return results
        
    except Exception as e:
        logger.error(f"传统回测失败: {e}")
        raise

def run_rl_model_backtest(df_features: pd.DataFrame, model: object, 
                         config: dict, test_period: str = "last_30_days") -> dict:
    """运行强化学习模型回测
    
    Args:
        df_features: 包含特征的DataFrame
        model: 训练好的强化学习模型
        config: 配置字典
        test_period: 测试期间 ("last_30_days", "last_60_days", "all")
        
    Returns:
        回测结果字典
    """
    logger.info("=" * 50)
    logger.info("运行强化学习模型回测...")
    logger.info("=" * 50)
    
    try:
        # 选择测试数据
        if test_period == "last_30_days":
            test_df = df_features.tail(30)
        elif test_period == "last_60_days":
            test_df = df_features.tail(60)
        else:
            test_df = df_features
        
        logger.info(f"测试数据期间: {test_df['date'].min()} 到 {test_df['date'].max()}")
        logger.info(f"测试数据样本数: {len(test_df)}")
        
        # 创建测试环境
        from src.models.trading_env import OilTradingEnv
        test_env = OilTradingEnv(test_df, "config/config.yaml")
        
        # 运行模型回测
        obs = test_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        actions = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward
            step_count += 1
            actions.append(action)
        
        # 获取投资组合统计
        portfolio_stats = test_env.get_portfolio_stats()
        
        # 构建回测结果
        rl_results = {
            'test_period': test_period,
            'test_start_date': test_df['date'].min(),
            'test_end_date': test_df['date'].max(),
            'total_reward': total_reward,
            'step_count': step_count,
            'portfolio_stats': portfolio_stats,
            'trades': test_env.trades,
            'actions': actions
        }
        
        # 保存回测结果
        results_dir = "results/rl_model_backtest"
        os.makedirs(results_dir, exist_ok=True)
        
        import json
        with open(f"{results_dir}/rl_backtest_results.json", 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型
            rl_results_serializable = {}
            for key, value in rl_results.items():
                if isinstance(value, np.ndarray):
                    rl_results_serializable[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    rl_results_serializable[key] = value.item()
                else:
                    rl_results_serializable[key] = value
            
            json.dump(rl_results_serializable, f, ensure_ascii=False, indent=2)
        
        logger.info(f"RL模型回测结果已保存到: {results_dir}/rl_backtest_results.json")
        
        # 输出回测结果摘要
        logger.info("RL模型回测结果摘要:")
        logger.info(f"  测试期间: {test_period}")
        logger.info(f"  总奖励: {total_reward:.4f}")
        logger.info(f"  总收益率: {portfolio_stats.get('total_return', 0):.4f}")
        logger.info(f"  夏普比率: {portfolio_stats.get('sharpe_ratio', 0):.4f}")
        logger.info(f"  最大回撤: {portfolio_stats.get('max_drawdown', 0):.4f}")
        logger.info(f"  总交易次数: {portfolio_stats.get('total_trades', 0)}")
        
        return rl_results
        
    except Exception as e:
        logger.error(f"RL模型回测失败: {e}")
        raise

def compare_strategies(df_features: pd.DataFrame, model: object, 
                      config: dict) -> dict:
    """比较传统策略和强化学习模型
    
    Args:
        df_features: 包含特征的DataFrame
        model: 训练好的强化学习模型
        config: 配置字典
        
    Returns:
        比较结果字典
    """
    logger.info("=" * 50)
    logger.info("比较传统策略和强化学习模型...")
    logger.info("=" * 50)
    
    try:
        # 运行传统策略回测
        traditional_results = run_traditional_backtest(df_features, config)
        
        # 运行RL模型回测
        rl_results = run_rl_model_backtest(df_features, model, config)
        
        # 构建比较结果
        comparison = {
            'traditional_strategies': traditional_results,
            'rl_model': rl_results,
            'comparison_date': datetime.now().isoformat()
        }
        
        # 保存比较结果
        results_dir = "results/strategy_comparison"
        os.makedirs(results_dir, exist_ok=True)
        
        import json
        with open(f"{results_dir}/strategy_comparison.json", 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        logger.info(f"策略比较结果已保存到: {results_dir}/strategy_comparison.json")
        
        # 输出比较摘要
        logger.info("策略比较摘要:")
        
        # 传统策略结果
        if 'results' in traditional_results:
            for strategy_name, results in traditional_results['results'].items():
                if results and 'portfolio_stats' in results:
                    stats = results['portfolio_stats']
                    logger.info(f"  {strategy_name}: 收益率={stats.get('total_return', 0):.4f}, "
                              f"夏普比率={stats.get('sharpe_ratio', 0):.4f}")
        
        # RL模型结果
        rl_stats = rl_results.get('portfolio_stats', {})
        logger.info(f"  RL模型: 收益率={rl_stats.get('total_return', 0):.4f}, "
                   f"夏普比率={rl_stats.get('sharpe_ratio', 0):.4f}")
        
        return comparison
        
    except Exception as e:
        logger.error(f"策略比较失败: {e}")
        raise

if __name__ == "__main__":
    """独立运行回测集成模块"""
    try:
        # 创建日志目录
        os.makedirs('logs', exist_ok=True)
        
        # 加载配置
        config = load_config()
        
        # 加载已处理的数据
        df_features, X, y = load_processed_data()
        
        # 加载训练好的模型
        model = load_trained_model()
        
        # 运行策略比较
        comparison = compare_strategies(df_features, model, config)
        
        logger.info("回测集成模块运行完成！")
        
    except Exception as e:
        logger.error(f"回测集成模块运行失败: {e}")
        raise 