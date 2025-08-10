#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国原油期货量化交易系统主程序
协调数据处理、模型训练和回测集成三个模块
"""

import os
import sys
import logging
import yaml
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入三个主要模块
from data_processing import process_data, load_processed_data
from model_training import train_model
from backtest_integration import run_backtest, prepare_backtest_data


# 设置matplotlib中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    try:
        from src.utils.config_manager import ConfigManager
        config_manager = ConfigManager(config_path)
        config = config_manager.processed_config
        logger.info("配置文件加载成功")
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        raise

def run_data_processing(config: dict, force_reprocess: bool = False) -> tuple:
    """运行数据处理模块"""
    logger.info("=" * 50)
    logger.info("运行数据处理模块...")
    logger.info("=" * 50)
    
    try:
        # 检查是否已有处理好的数据
        if not force_reprocess and os.path.exists("data/processed/features_data.csv"):
            logger.info("发现已处理的数据，直接加载...")
            df_features, X, y = load_processed_data()
            return df_features, X, y, None
        else:
            logger.info("开始处理数据...")
            df_features, X, y, engineer = process_data(config, save_processed=True)
            return df_features, X, y, engineer
            
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        raise

def run_model_training(df_features: pd.DataFrame, config: dict, force_retrain: bool = False) -> dict:
    """运行模型训练模块"""
    logger.info("=" * 50)
    logger.info("运行模型训练模块...")
    logger.info("=" * 50)
    
    try:
        # 检查是否已有训练好的模型
        if not force_retrain and os.path.exists("models/training_results/training_config.json"):
            logger.info("发现已训练的模型，跳过训练...")
            return {"status": "model_exists", "message": "模型已存在"}
        else:
            logger.info("开始训练模型...")
            results = train_model(config, df_features, save_model=True)
            return results
            
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise

def run_backtest_integration(df_features: pd.DataFrame, config: dict) -> dict:
    """运行回测集成模块"""
    logger.info("=" * 50)
    logger.info("运行回测集成模块...")
    logger.info("=" * 50)
    
    try:
        # 加载训练好的模型
        model = load_trained_model()
        
        # 运行策略比较
        comparison = compare_strategies(df_features, model, config)
        
        return comparison
        
    except Exception as e:
        logger.error(f"回测集成失败: {e}")
        raise



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中国原油期货量化交易系统')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')

    parser.add_argument('--force-reprocess', action='store_true', help='强制重新处理数据')
    parser.add_argument('--force-retrain', action='store_true', help='强制重新训练模型')
    parser.add_argument('--no-data', action='store_true', help='跳过数据处理')
    parser.add_argument('--no-train', action='store_true', help='跳过模型训练')
    parser.add_argument('--no-backtest', action='store_true', help='跳过回测集成')
    
    args = parser.parse_args()
    
    try:
        # 创建日志目录
        os.makedirs('logs', exist_ok=True)
        
        # 加载配置
        config = load_config(args.config)
        
        # 1. 数据处理
        df_features, X, y, engineer = None, None, None, None
        if not args.no_data:
            df_features, X, y, engineer = run_data_processing(config, force_reprocess=args.force_reprocess)
        
        # 2. 模型训练
        training_results = None
        if not args.no_train and df_features is not None:
            training_results = run_model_training(df_features, config, force_retrain=args.force_retrain)
        
        # 3. 回测集成
        backtest_results = None
        if not args.no_backtest and df_features is not None:
            backtest_results = run_backtest_integration(df_features, config)
        
        # 输出总结
        logger.info("=" * 50)
        logger.info("系统运行完成！")
        logger.info("=" * 50)
        
        if training_results:
            logger.info("模型训练结果:")
            if training_results.get("status") == "model_exists":
                logger.info("  模型已存在，跳过训练")
            elif 'evaluation_results' in training_results:
                stats = training_results['evaluation_results']['portfolio_stats']
                logger.info(f"  总收益率: {stats.get('total_return', 0):.4f}")
                logger.info(f"  夏普比率: {stats.get('sharpe_ratio', 0):.4f}")
                logger.info(f"  最大回撤: {stats.get('max_drawdown', 0):.4f}")
                logger.info(f"  总交易次数: {stats.get('total_trades', 0)}")
            

        
        if backtest_results:
            logger.info("回测集成结果:")
            if 'traditional_strategies' in backtest_results:
                logger.info("  传统策略结果:")
                traditional = backtest_results['traditional_strategies']
                if 'results' in traditional:
                    for strategy_name, results in traditional['results'].items():
                        if results and 'portfolio_stats' in results:
                            stats = results['portfolio_stats']
                            logger.info(f"    {strategy_name}: 收益率={stats.get('total_return', 0):.4f}, "
                                      f"夏普比率={stats.get('sharpe_ratio', 0):.4f}")
            
            if 'rl_model' in backtest_results:
                logger.info("  强化学习模型结果:")
                rl_stats = backtest_results['rl_model']['portfolio_stats']
                logger.info(f"    RL模型: 收益率={rl_stats.get('total_return', 0):.4f}, "
                           f"夏普比率={rl_stats.get('sharpe_ratio', 0):.4f}")
        
        logger.info(f"结果文件保存在: {config['storage']['results_path']}")
        logger.info(f"模型文件保存在: {config['storage']['model_path']}")
        
    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        raise

if __name__ == "__main__":
    main() 