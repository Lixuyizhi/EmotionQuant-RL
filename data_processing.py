#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块
负责数据加载、特征工程和数据准备
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

from src.data.data_loader import OilDataLoader
from src.features.feature_engineering import FeatureEngineer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_processing.log', encoding='utf-8'),
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

def process_data(config: dict, save_processed: bool = True) -> tuple:
    """数据处理主函数
    
    Args:
        config: 配置字典
        save_processed: 是否保存处理后的数据
        
    Returns:
        df_features: 包含特征的DataFrame
        X: 特征矩阵
        y: 目标变量
        engineer: 特征工程器实例
    """
    logger.info("=" * 50)
    logger.info("开始数据处理...")
    logger.info("=" * 50)
    
    try:
        # 1. 加载原始数据
        logger.info("1. 加载原始数据...")
        loader = OilDataLoader()
        df = loader.get_data_with_cache(use_cache=True)
        
        # 验证数据
        if not loader.validate_data(df):
            raise ValueError("数据验证失败")
        
        logger.info(f"原始数据加载完成，数据形状: {df.shape}")
        logger.info(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
        
        # 2. 特征工程
        logger.info("2. 进行特征工程...")
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
        
        # 3. 准备机器学习特征
        logger.info("3. 准备机器学习特征...")
        X, y = engineer.prepare_features_for_ml(df_features)
        
        logger.info(f"特征矩阵形状: {X.shape}")
        logger.info(f"目标变量形状: {y.shape}")
        logger.info(f"特征列名: {list(X.columns)}")
        
        # 4. 保存处理后的数据
        if save_processed:
            logger.info("4. 保存处理后的数据...")
            data_dir = "data/processed"
            os.makedirs(data_dir, exist_ok=True)
            
            # 保存完整特征数据
            df_features.to_csv(f"{data_dir}/features_data.csv", index=False)
            logger.info(f"特征数据已保存到: {data_dir}/features_data.csv")
            
            # 保存特征矩阵和目标变量
            X.to_csv(f"{data_dir}/features_matrix.csv", index=False)
            y.to_csv(f"{data_dir}/target_variable.csv", index=False)
            logger.info(f"特征矩阵已保存到: {data_dir}/features_matrix.csv")
            logger.info(f"目标变量已保存到: {data_dir}/target_variable.csv")
            
            # 保存特征信息
            feature_info = {
                'feature_count': X.shape[1],
                'sample_count': X.shape[0],
                'feature_names': list(X.columns),
                'processing_date': datetime.now().isoformat(),
                'data_range': {
                    'start_date': df['date'].min().isoformat(),
                    'end_date': df['date'].max().isoformat()
                }
            }
            
            import json
            with open(f"{data_dir}/feature_info.json", 'w', encoding='utf-8') as f:
                json.dump(feature_info, f, ensure_ascii=False, indent=2)
            logger.info(f"特征信息已保存到: {data_dir}/feature_info.json")
        
        logger.info("=" * 50)
        logger.info("数据处理完成！")
        logger.info("=" * 50)
        
        return df_features, X, y, engineer
        
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        raise

def load_processed_data(data_dir: str = "data/processed") -> tuple:
    """加载已处理的数据
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        df_features: 包含特征的DataFrame
        X: 特征矩阵
        y: 目标变量
    """
    logger.info(f"加载已处理的数据从: {data_dir}")
    
    try:
        # 加载特征数据
        df_features = pd.read_csv(f"{data_dir}/features_data.csv")
        X = pd.read_csv(f"{data_dir}/features_matrix.csv")
        y = pd.read_csv(f"{data_dir}/target_variable.csv").squeeze()
        
        logger.info(f"数据加载完成，特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
        
        return df_features, X, y
        
    except Exception as e:
        logger.error(f"加载已处理数据失败: {e}")
        raise

if __name__ == "__main__":
    """独立运行数据处理模块"""
    try:
        # 创建日志目录
        os.makedirs('logs', exist_ok=True)
        
        # 加载配置
        config = load_config()
        
        # 处理数据
        df_features, X, y, engineer = process_data(config, save_processed=True)
        
        logger.info("数据处理模块运行完成！")
        
    except Exception as e:
        logger.error(f"数据处理模块运行失败: {e}")
        raise 