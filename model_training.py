#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练模块
负责强化学习模型训练和保存
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

from src.models.rl_trainer import RLTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log', encoding='utf-8'),
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

def load_processed_data(data_dir: str = "data/processed") -> tuple:
    """加载已处理的数据"""
    logger.info(f"加载已处理的数据从: {data_dir}")
    
    try:
        df_features = pd.read_csv(f"{data_dir}/features_data.csv")
        X = pd.read_csv(f"{data_dir}/features_matrix.csv")
        y = pd.read_csv(f"{data_dir}/target_variable.csv").squeeze()
        
        logger.info(f"数据加载完成，特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
        
        return df_features, X, y
        
    except Exception as e:
        logger.error(f"加载已处理数据失败: {e}")
        raise

def train_model(config: dict, df_features: pd.DataFrame, 
                algorithm: str = None, total_timesteps: int = None,
                env_type: str = None, save_model: bool = True) -> dict:
    """训练强化学习模型
    
    Args:
        config: 配置字典
        df_features: 包含特征的DataFrame
        algorithm: 算法名称，默认使用配置文件中的设置
        total_timesteps: 训练步数，默认使用配置文件中的设置
        env_type: 环境类型，若为None则自动从config['model_training']['model']['env_name']读取
        save_model: 是否保存模型
        
    Returns:
        训练结果字典
    """
    logger.info("=" * 50)
    logger.info("开始模型训练...")
    logger.info("=" * 50)
    
    try:
        # 使用配置中的默认值
        algorithm = algorithm or config['model_training']['model']['algorithm']
        total_timesteps = total_timesteps or config['model_training']['model']['total_timesteps']
        env_name = env_type or config['model_training']['model'].get('env_name', 'SignalWeightTradingEnv')
        logger.info(f"使用环境: {env_name}")
        
        logger.info(f"训练算法: {algorithm}")
        logger.info(f"训练步数: {total_timesteps}")
        
        # 创建训练器
        trainer = RLTrainer()
        
        # 训练模型
        logger.info("开始训练...")
        results = trainer.train_model(
            df=df_features,
            env_type=env_name,
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            config=config
        )
        
        # 保存训练结果
        if save_model:
            logger.info("保存训练结果...")
            results_dir = "models/training_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # 保存训练配置
            portfolio_stats = results.get('evaluation_results', {}).get('portfolio_stats', {})
            
            # 转换numpy类型为Python原生类型
            portfolio_stats_serializable = {}
            for key, value in portfolio_stats.items():
                if isinstance(value, np.ndarray):
                    portfolio_stats_serializable[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    portfolio_stats_serializable[key] = value.item()
                else:
                    portfolio_stats_serializable[key] = value
            
            training_config = {
                'algorithm': algorithm,
                'total_timesteps': total_timesteps,
                'training_date': datetime.now().isoformat(),
                'model_path': results.get('model_path', ''),
                'portfolio_stats': portfolio_stats_serializable
            }
            
            import json
            with open(f"{results_dir}/training_config.json", 'w', encoding='utf-8') as f:
                json.dump(training_config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"训练配置已保存到: {results_dir}/training_config.json")
            
            # 保存评估结果
            if 'evaluation_results' in results:
                eval_results = results['evaluation_results']
                eval_file = f"{results_dir}/evaluation_results.json"
                
                # 递归转换所有numpy类型为Python原生类型
                def to_serializable(val):
                    if isinstance(val, np.ndarray):
                        return val.tolist()
                    elif isinstance(val, (np.integer, np.floating)):
                        return val.item()
                    elif isinstance(val, dict):
                        return {k: to_serializable(v) for k, v in val.items()}
                    elif isinstance(val, list):
                        return [to_serializable(v) for v in val]
                    else:
                        return val
                
                with open(eval_file, 'w', encoding='utf-8') as f:
                    json.dump(to_serializable(eval_results), f, ensure_ascii=False, indent=2)
                
                logger.info(f"评估结果已保存到: {eval_file}")
        
        logger.info("=" * 50)
        logger.info("模型训练完成！")
        logger.info("=" * 50)
        
        # 输出训练结果摘要
        if 'evaluation_results' in results and 'portfolio_stats' in results['evaluation_results']:
            stats = results['evaluation_results']['portfolio_stats']
            logger.info("训练结果摘要:")
            logger.info(f"  总收益率: {stats.get('total_return', 0):.4f}")
            logger.info(f"  夏普比率: {stats.get('sharpe_ratio', 0):.4f}")
            logger.info(f"  最大回撤: {stats.get('max_drawdown', 0):.4f}")
            logger.info(f"  总交易次数: {stats.get('total_trades', 0)}")
        
        return results
        
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise

def compare_algorithms(config: dict, df_features: pd.DataFrame, 
                      algorithms: list = None, total_timesteps: int = None) -> dict:
    """比较不同算法的性能
    
    Args:
        config: 配置字典
        df_features: 包含特征的DataFrame
        algorithms: 要比较的算法列表
        total_timesteps: 训练步数
        
    Returns:
        比较结果字典
    """
    logger.info("=" * 50)
    logger.info("开始算法比较...")
    logger.info("=" * 50)
    
    algorithms = algorithms or ["PPO", "A2C", "DQN"]
    total_timesteps = total_timesteps or config['model_training']['model']['total_timesteps']
    
    comparison_results = {}
    
    for algorithm in algorithms:
        logger.info(f"训练 {algorithm} 算法...")
        try:
            results = train_model(config, df_features, algorithm, total_timesteps, save_model=False)
            comparison_results[algorithm] = results
            logger.info(f"{algorithm} 训练完成")
        except Exception as e:
            logger.error(f"{algorithm} 训练失败: {e}")
            comparison_results[algorithm] = None
    
    # 保存比较结果
    results_dir = "models/algorithm_comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    comparison_summary = {}
    for algorithm, results in comparison_results.items():
        if results and 'evaluation_results' in results:
            stats = results['evaluation_results']['portfolio_stats']
            comparison_summary[algorithm] = {
                'total_return': stats.get('total_return', 0),
                'sharpe_ratio': stats.get('sharpe_ratio', 0),
                'max_drawdown': stats.get('max_drawdown', 0),
                'total_trades': stats.get('total_trades', 0)
            }
    
    import json
    with open(f"{results_dir}/comparison_summary.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"算法比较结果已保存到: {results_dir}/comparison_summary.json")
    
    # 输出最佳算法
    if comparison_summary:
        best_algorithm = max(comparison_summary.items(), 
                           key=lambda x: x[1]['sharpe_ratio'])
        logger.info(f"最佳算法: {best_algorithm[0]} (夏普比率: {best_algorithm[1]['sharpe_ratio']:.4f})")
    
    return comparison_results

if __name__ == "__main__":
    """独立运行模型训练模块"""
    try:
        # 创建日志目录
        os.makedirs('logs', exist_ok=True)
        
        # 加载配置
        config = load_config()
        
        # 加载已处理的数据
        df_features, X, y = load_processed_data()
        
        # 训练模型
        results = train_model(config, df_features)
        
        logger.info("模型训练模块运行完成！")
        
    except Exception as e:
        logger.error(f"模型训练模块运行失败: {e}")
        raise 