#!/usr/bin/env python3
"""
测试OHLC特征是否正确添加到观察空间中
"""

import numpy as np
import pandas as pd
from src.data.data_loader import OilDataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.signal_weight_env import SignalWeightTradingEnv
from src.models.max_weight_env import MaxWeightTradingEnv

def test_ohlc_features():
    """测试OHLC特征"""
    print("=== 测试OHLC特征添加 ===")
    
    # 1. 加载数据
    print("1. 加载数据...")
    loader = OilDataLoader()
    df = loader.get_data_with_cache(use_cache=True)
    print(f"   原始数据形状: {df.shape}")
    
    # 2. 特征工程
    print("2. 特征工程...")
    engineer = FeatureEngineer()
    df_features = engineer.add_technical_indicators(df)
    df_features = engineer.add_target_variables(df_features)
    df_features = engineer.create_trading_features(df_features)
    print(f"   特征工程后数据形状: {df_features.shape}")
    
    # 3. 检查OHLC特征
    print("3. 检查OHLC特征...")
    ohlc_cols = [col for col in df_features.columns if col.startswith('norm_') or col.startswith('zscore_')]
    print(f"   OHLC特征列: {ohlc_cols}")
    
    # 检查前几行的OHLC数据
    print("   前5行OHLC数据:")
    for col in ['norm_open', 'norm_high', 'norm_low', 'norm_close']:
        if col in df_features.columns:
            values = df_features[col].head()
            print(f"   {col}: {values.values}")
    
    # 4. 测试SignalWeightTradingEnv
    print("\n4. 测试SignalWeightTradingEnv...")
    env = SignalWeightTradingEnv(df_features)
    obs, info = env.reset()
    print(f"   观察空间形状: {env.observation_space.shape}")
    print(f"   观察向量长度: {len(obs)}")
    print(f"   观察向量: {obs}")
    print(f"   OHLC部分 (索引6-9): {obs[6:10]}")
    
    # 5. 测试MaxWeightTradingEnv
    print("\n5. 测试MaxWeightTradingEnv...")
    env2 = MaxWeightTradingEnv(df_features)
    obs2, info2 = env2.reset()
    print(f"   观察空间形状: {env2.observation_space.shape}")
    print(f"   观察向量长度: {len(obs2)}")
    print(f"   观察向量: {obs2}")
    print(f"   OHLC部分 (索引6-9): {obs2[6:10]}")
    
    # 6. 测试环境步骤
    print("\n6. 测试环境步骤...")
    action = np.array([0.3, 0.4, 0.3])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   步骤执行成功!")
    print(f"   奖励: {reward}")
    print(f"   新的OHLC数据: {obs[6:10]}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_ohlc_features() 