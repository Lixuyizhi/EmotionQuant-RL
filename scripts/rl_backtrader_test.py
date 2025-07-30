#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习模型Backtrader回测脚本
在Backtrader框架中集成训练好的强化学习模型进行回测
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.backtrader_strategies import BacktraderBacktester, RLModelStrategy
from src.data.data_loader import OilDataLoader
from src.features.feature_engineering import FeatureEngineer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def prepare_data():
    """准备回测数据"""
    logger.info("准备回测数据...")
    
    # 加载数据
    loader = OilDataLoader()
    df = loader.get_data_with_cache(use_cache=True)
    
    # 特征工程
    engineer = FeatureEngineer()
    df_features = engineer.add_technical_indicators(df)
    df_features = engineer.add_target_variables(df_features)
    df_features = engineer.create_trading_features(df_features)
    
    logger.info(f"数据准备完成，数据形状: {df_features.shape}")
    return df_features

def run_rl_model_backtest(df_features: pd.DataFrame, model_path: str, 
                         algorithm: str = 'A2C', env_type: str = 'signal_weight_env'):
    """运行RL模型Backtrader回测"""
    logger.info("=" * 50)
    logger.info("开始RL模型Backtrader回测...")
    logger.info("=" * 50)
    
    try:
        # 创建回测器
        backtester = BacktraderBacktester()
        
        # 运行RL模型策略
        rl_results = backtester.run_backtest(
            df=df_features,
            strategy_class=RLModelStrategy,
            strategy_name="RL_Model_Strategy",
            model_path=model_path,
            algorithm=algorithm,
            env_type=env_type
        )
        
        logger.info("RL模型回测完成!")
        return rl_results
        
    except Exception as e:
        logger.error(f"RL模型回测失败: {e}")
        raise

def run_traditional_strategies_backtest(df_features: pd.DataFrame):
    """运行传统策略回测作为对比"""
    logger.info("=" * 50)
    logger.info("运行传统策略回测...")
    logger.info("=" * 50)
    
    try:
        # 创建回测器
        backtester = BacktraderBacktester()
        
        # 运行传统策略比较
        traditional_results = backtester.compare_strategies(df_features)
        
        logger.info("传统策略回测完成!")
        return traditional_results
        
    except Exception as e:
        logger.error(f"传统策略回测失败: {e}")
        raise

def compare_results(rl_results: dict, traditional_results: dict):
    """比较RL模型和传统策略的结果"""
    logger.info("=" * 50)
    logger.info("结果比较分析...")
    logger.info("=" * 50)
    
    # 提取RL模型结果
    rl_stats = rl_results.get('portfolio_stats', {})
    rl_analyzers = rl_results.get('analyzers', {})
    
    # 提取传统策略结果
    traditional_stats = {}
    for strategy_name, result in traditional_results.get('results', {}).items():
        if result:
            traditional_stats[strategy_name] = {
                'total_return': result.get('total_return', 0),
                'sharpe_ratio': result.get('analyzers', {}).get('sharpe_ratio', {}).get('sharperatio', 0),
                'max_drawdown': result.get('analyzers', {}).get('drawdown', {}).get('max', {}).get('drawdown', 0),
                'final_value': result.get('final_value', 0)
            }
    
    # 打印比较结果
    print("\n" + "="*60)
    print("RL模型 vs 传统策略性能比较")
    print("="*60)
    
    print(f"\nRL模型策略:")
    print(f"  总收益率: {rl_stats.get('total_return', 0):.4f}")
    print(f"  夏普比率: {rl_analyzers.get('sharpe_ratio', {}).get('sharperatio', 0):.4f}")
    print(f"  最大回撤: {rl_analyzers.get('drawdown', {}).get('max', {}).get('drawdown', 0):.4f}")
    print(f"  最终价值: {rl_results.get('final_value', 0):.2f}")
    
    print(f"\n传统策略:")
    for strategy_name, stats in traditional_stats.items():
        print(f"  {strategy_name}:")
        print(f"    总收益率: {stats['total_return']:.4f}")
        print(f"    夏普比率: {stats['sharpe_ratio']:.4f}")
        print(f"    最大回撤: {stats['max_drawdown']:.4f}")
        print(f"    最终价值: {stats['final_value']:.2f}")
    
    # 找出最佳策略
    all_strategies = {'RL_Model': {
        'total_return': rl_stats.get('total_return', 0),
        'sharpe_ratio': rl_analyzers.get('sharpe_ratio', {}).get('sharperatio', 0),
        'max_drawdown': rl_analyzers.get('drawdown', {}).get('max', {}).get('drawdown', 0)
    }}
    all_strategies.update(traditional_stats)
    
    best_return = max(all_strategies.items(), key=lambda x: x[1]['total_return'])
    best_sharpe = max(all_strategies.items(), key=lambda x: x[1]['sharpe_ratio'])
    
    print(f"\n最佳策略:")
    print(f"  最高收益率: {best_return[0]} ({best_return[1]['total_return']:.4f})")
    print(f"  最高夏普比率: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.4f})")

def plot_comparison_results(rl_results: dict, traditional_results: dict, save_path: str = None):
    """绘制比较结果"""
    if save_path is None:
        save_path = "results/rl_vs_traditional_comparison.png"
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 准备数据
    strategies = ['RL_Model']
    returns = [rl_results.get('total_return', 0)]
    sharpe_ratios = [rl_results.get('analyzers', {}).get('sharpe_ratio', {}).get('sharperatio', 0)]
    drawdowns = [rl_results.get('analyzers', {}).get('drawdown', {}).get('max', {}).get('drawdown', 0)]
    
    # 添加传统策略数据
    for strategy_name, result in traditional_results.get('results', {}).items():
        if result:
            strategies.append(strategy_name)
            returns.append(result.get('total_return', 0))
            sharpe_ratios.append(result.get('analyzers', {}).get('sharpe_ratio', {}).get('sharperatio', 0))
            drawdowns.append(result.get('analyzers', {}).get('drawdown', {}).get('max', {}).get('drawdown', 0))
    
    # 绘制收益率比较
    axes[0, 0].bar(strategies, returns, color=['red'] + ['blue'] * (len(strategies)-1))
    axes[0, 0].set_title('总收益率比较')
    axes[0, 0].set_ylabel('总收益率')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 绘制夏普比率比较
    axes[0, 1].bar(strategies, sharpe_ratios, color=['red'] + ['blue'] * (len(strategies)-1))
    axes[0, 1].set_title('夏普比率比较')
    axes[0, 1].set_ylabel('夏普比率')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 绘制最大回撤比较
    axes[1, 0].bar(strategies, drawdowns, color=['red'] + ['blue'] * (len(strategies)-1))
    axes[1, 0].set_title('最大回撤比较')
    axes[1, 0].set_ylabel('最大回撤')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 绘制投资组合价值曲线
    # RL模型价值曲线
    if 'portfolio_values' in rl_results:
        rl_values = rl_results['portfolio_values']
        dates = pd.date_range(start=datetime.now(), periods=len(rl_values), freq='D')
        axes[1, 1].plot(dates, rl_values, label='RL_Model', color='red', linewidth=2)
    
    # 传统策略价值曲线
    for strategy_name, result in traditional_results.get('results', {}).items():
        if result and 'portfolio_values' in result:
            values = result['portfolio_values']
            dates = pd.date_range(start=datetime.now(), periods=len(values), freq='D')
            axes[1, 1].plot(dates, values, label=strategy_name, alpha=0.7)
    
    axes[1, 1].set_title('投资组合价值曲线')
    axes[1, 1].set_ylabel('投资组合价值')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"比较结果图表已保存到: {save_path}")

def main():
    """主函数"""
    # 配置参数
    model_path = "models/A2C_SignalWeightTradingEnv_model.zip"  # 模型路径
    algorithm = "A2C"  # 算法类型
    env_type = "signal_weight_env"  # 环境类型
    
    try:
        # 1. 准备数据
        df_features = prepare_data()
        
        # 2. 运行RL模型回测
        rl_results = run_rl_model_backtest(df_features, model_path, algorithm, env_type)
        
        # 3. 运行传统策略回测
        traditional_results = run_traditional_strategies_backtest(df_features)
        
        # 4. 比较结果
        compare_results(rl_results, traditional_results)
        
        # 5. 绘制比较图表
        plot_comparison_results(rl_results, traditional_results)
        
        logger.info("=" * 50)
        logger.info("RL模型Backtrader回测完成!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"回测失败: {e}")
        raise

if __name__ == "__main__":
    main() 