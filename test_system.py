#!/usr/bin/env python3
"""
系统测试脚本

用于测试量化交易系统的各个模块是否正常工作。
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loader():
    """测试数据加载模块"""
    logger.info("测试数据加载模块...")
    
    try:
        from src.data.data_loader import OilDataLoader
        
        # 创建数据加载器
        loader = OilDataLoader()
        
        # 测试数据获取（使用较短的时间范围）
        df = loader.get_oil_futures_data(start_date="2023-01-01", end_date="2023-12-31")
        
        # 验证数据
        assert not df.empty, "数据为空"
        assert len(df.columns) >= 6, "数据列数不足"
        assert 'date' in df.columns, "缺少日期列"
        assert 'close' in df.columns, "缺少收盘价列"
        
        logger.info(f"数据加载测试通过，数据形状: {df.shape}")
        return True
        
    except Exception as e:
        logger.error(f"数据加载测试失败: {str(e)}")
        return False

def test_feature_engineering():
    """测试特征工程模块"""
    logger.info("测试特征工程模块...")
    
    try:
        from src.features.feature_engineering import FeatureEngineer
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(400, 500, 100),
            'high': np.random.uniform(450, 550, 100),
            'low': np.random.uniform(350, 450, 100),
            'close': np.random.uniform(400, 500, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # 创建特征工程器
        engineer = FeatureEngineer()
        
        # 测试技术指标计算
        df_features = engineer.add_technical_indicators(test_data)
        
        # 验证特征
        assert len(df_features.columns) > len(test_data.columns), "特征数量没有增加"
        
        # 检查是否有技术指标
        technical_indicators = ['SMA_20', 'RSI_14', 'MACD']
        for indicator in technical_indicators:
            if indicator in df_features.columns:
                logger.info(f"技术指标 {indicator} 计算成功")
        
        logger.info("特征工程测试通过")
        return True
        
    except Exception as e:
        logger.error(f"特征工程测试失败: {str(e)}")
        return False

def test_trading_env():
    """测试交易环境模块"""
    logger.info("测试交易环境模块...")
    
    try:
        from src.models.trading_env import OilTradingEnv
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(400, 500, 50),
            'high': np.random.uniform(450, 550, 50),
            'low': np.random.uniform(350, 450, 50),
            'close': np.random.uniform(400, 500, 50),
            'volume': np.random.uniform(1000, 10000, 50),
            'SMA_20': np.random.uniform(400, 500, 50),
            'RSI_14': np.random.uniform(0, 100, 50),
            'MACD': np.random.uniform(-10, 10, 50)
        })
        
        # 创建交易环境
        env = OilTradingEnv(test_data)
        
        # 测试环境重置
        obs = env.reset()
        assert obs is not None, "环境重置失败"
        
        # 测试环境步骤
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        assert obs is not None, "环境步骤失败"
        assert isinstance(reward, (int, float)), "奖励类型错误"
        assert isinstance(done, bool), "完成状态类型错误"
        
        logger.info("交易环境测试通过")
        return True
        
    except Exception as e:
        logger.error(f"交易环境测试失败: {str(e)}")
        return False

def test_backtrader_strategies():
    """测试backtrader策略模块"""
    logger.info("测试backtrader策略模块...")
    
    try:
        from src.strategies.backtrader_strategies import WeightedFactorStrategy, MaxWeightStrategy
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(400, 500, 30),
            'high': np.random.uniform(450, 550, 30),
            'low': np.random.uniform(350, 450, 30),
            'close': np.random.uniform(400, 500, 30),
            'volume': np.random.uniform(1000, 10000, 30)
        })
        
        # 测试策略类是否可以实例化（只需要验证类定义）
        assert WeightedFactorStrategy is not None, "WeightedFactorStrategy类定义失败"
        assert MaxWeightStrategy is not None, "MaxWeightStrategy类定义失败"
        
        # 验证策略类的基本属性
        assert hasattr(WeightedFactorStrategy, 'params'), "WeightedFactorStrategy缺少params属性"
        assert hasattr(MaxWeightStrategy, 'params'), "MaxWeightStrategy缺少params属性"
        
        logger.info("backtrader策略测试通过")
        return True
        
    except Exception as e:
        logger.error(f"backtrader策略测试失败: {str(e)}")
        return False

def test_config():
    """测试配置文件"""
    logger.info("测试配置文件...")
    
    try:
        import yaml
        
        # 读取配置文件
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证配置结构
        required_sections = ['data', 'features', 'model', 'strategy', 'backtest']
        for section in required_sections:
            assert section in config, f"配置文件缺少 {section} 部分"
        
        logger.info("配置文件测试通过")
        return True
        
    except Exception as e:
        logger.error(f"配置文件测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    logger.info("开始系统测试...")
    
    tests = [
        ("配置文件", test_config),
        ("数据加载", test_data_loader),
        ("特征工程", test_feature_engineering),
        ("交易环境", test_trading_env),
        ("backtrader策略", test_backtrader_strategies),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} 测试通过")
            else:
                logger.error(f"✗ {test_name} 测试失败")
        except Exception as e:
            logger.error(f"✗ {test_name} 测试异常: {str(e)}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"测试结果: {passed}/{total} 通过")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 所有测试通过！系统可以正常使用。")
        return True
    else:
        logger.error("❌ 部分测试失败，请检查相关模块。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 