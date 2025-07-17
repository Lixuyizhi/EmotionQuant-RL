#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试信号权重环境
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.signal_weight_env import SignalWeightTradingEnv
from data_processing import load_processed_data

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_signal_weight_env():
    """测试信号权重环境"""
    logger.info("=" * 50)
    logger.info("测试信号权重环境")
    logger.info("=" * 50)
    
    try:
        # 加载已处理的数据
        df_features, X, y = load_processed_data()
        logger.info(f"数据加载完成，特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
        
        # 创建信号权重环境
        env = SignalWeightTradingEnv(df_features)
        logger.info("信号权重环境创建成功")
        
        # 打印环境信息
        logger.info(f"观察空间形状: {env.observation_space.shape}")
        logger.info(f"动作空间形状: {env.action_space.shape}")
        logger.info(f"交易信号列: {env.signal_columns}")
        logger.info("=== 环境参数配置 ===")
        logger.info(f"初始余额: {env.initial_balance}")
        logger.info(f"手续费率: {env.transaction_fee:.4f}")
        logger.info(f"滑点率: {env.slippage:.4f}")
        logger.info(f"仓位大小: {env.position_size:.2f}")
        logger.info(f"买入阈值: {env.buy_threshold}")
        logger.info(f"卖出阈值: {env.sell_threshold}")
        logger.info(f"最大仓位比例: {env.max_position_ratio}")
        logger.info(f"最小交易金额: {env.min_trade_amount}")
        logger.info(f"奖励缩放因子: {env.reward_scale}")
        logger.info(f"风险惩罚系数: {env.risk_penalty}")
        logger.info("==================")
        
        # 重置环境
        obs, info = env.reset()
        logger.info(f"环境重置完成，观察形状: {obs.shape}")
        
        # 测试几个步骤
        total_reward = 0
        for step in range(10):
            # 生成随机权重动作
            action = np.random.random(3)  # 3个权重值
            logger.info(f"步骤 {step + 1}: 动作权重 = {action}")
            
            # 执行步骤
            obs, reward, terminated, truncated, info = env.step(action)
            
            logger.info(f"  奖励: {reward:.4f}")
            logger.info(f"  加权信号: {info['weighted_signal']:.4f}")
            logger.info(f"  交易动作: {info['trade_action']} (-1=卖出, 0=持有, 1=买入)")
            logger.info(f"  归一化权重: {info['signal_weights']}")
            logger.info(f"  当前信号: {info['current_signals']}")
            logger.info(f"  组合价值: {info['portfolio_value']:.2f}")
            
            total_reward += reward
            
            if terminated:
                logger.info("环境结束")
                break
        
        logger.info(f"总奖励: {total_reward:.4f}")
        
        # 获取组合统计
        stats = env.get_portfolio_stats()
        logger.info("组合统计:")
        for key, value in stats.items():
            if isinstance(value, (list, np.ndarray)):
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value:.4f}")
        
        # 获取信号权重分析
        weights_analysis = env.get_signal_weights_analysis()
        logger.info("信号权重分析:")
        for key, value in weights_analysis.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("=" * 50)
        logger.info("信号权重环境测试完成！")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise

if __name__ == "__main__":
    test_signal_weight_env() 