#!/usr/bin/env python3
"""
快速演示脚本

运行一个简化的量化交易系统演示，展示主要功能。
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_data():
    """创建演示数据"""
    logger.info("创建演示数据...")
    
    # 生成模拟的原油期货数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
    # 模拟价格走势（带有趋势和波动）
    base_price = 450
    trend = np.linspace(0, 50, 200)  # 上升趋势
    noise = np.random.normal(0, 10, 200)  # 随机波动
    prices = base_price + trend + noise
    
    # 生成OHLCV数据
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # 生成开盘价、最高价、最低价
        daily_volatility = np.random.uniform(0.01, 0.03)
        open_price = close * (1 + np.random.normal(0, daily_volatility))
        high_price = max(open_price, close) * (1 + np.random.uniform(0, 0.02))
        low_price = min(open_price, close) * (1 - np.random.uniform(0, 0.02))
        volume = np.random.uniform(5000, 15000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    logger.info(f"演示数据创建完成，数据形状: {df.shape}")
    
    return df

def run_demo():
    """运行演示"""
    logger.info("开始量化交易系统演示...")
    
    try:
        # 1. 创建演示数据
        df = create_demo_data()
        
        # 2. 特征工程
        logger.info("进行特征工程...")
        from src.features.feature_engineering import FeatureEngineer
        
        engineer = FeatureEngineer()
        df_features = engineer.add_technical_indicators(df)
        df_features = engineer.add_target_variables(df_features)
        df_features = engineer.create_trading_features(df_features)
        
        logger.info(f"特征工程完成，特征数量: {len(df_features.columns) - 6}")
        
        # 3. 简单的策略回测（不使用backtrader，直接计算）
        logger.info("进行简单策略回测...")
        
        # 计算简单的移动平均策略
        df_features['SMA_20'] = df_features['close'].rolling(20).mean()
        df_features['SMA_50'] = df_features['close'].rolling(50).mean()
        
        # 生成交易信号
        df_features['signal'] = 0
        df_features.loc[df_features['SMA_20'] > df_features['SMA_50'], 'signal'] = 1  # 买入信号
        df_features.loc[df_features['SMA_20'] < df_features['SMA_50'], 'signal'] = -1  # 卖出信号
        
        # 计算策略收益
        initial_capital = 100000
        position = 0
        capital = initial_capital
        portfolio_values = []
        
        for i, row in df_features.iterrows():
            if pd.isna(row['SMA_20']) or pd.isna(row['SMA_50']):
                portfolio_values.append(capital)
                continue
            
            if row['signal'] == 1 and position == 0:  # 买入
                position = capital / row['close']
                capital = 0
            elif row['signal'] == -1 and position > 0:  # 卖出
                capital = position * row['close']
                position = 0
            
            # 计算当前投资组合价值
            current_value = capital + (position * row['close'])
            portfolio_values.append(current_value)
        
        # 4. 计算性能指标
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # 计算最大回撤
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        # 5. 显示结果
        logger.info("\n" + "="*50)
        logger.info("演示结果")
        logger.info("="*50)
        logger.info(f"初始资金: {initial_capital:,.2f}")
        logger.info(f"最终资金: {portfolio_values[-1]:,.2f}")
        logger.info(f"总收益率: {total_return:.4f} ({total_return*100:.2f}%)")
        logger.info(f"年化波动率: {volatility:.4f}")
        logger.info(f"夏普比率: {sharpe_ratio:.4f}")
        logger.info(f"最大回撤: {max_dd:.4f} ({max_dd*100:.2f}%)")
        logger.info("="*50)
        
        # 6. 绘制结果
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 价格和移动平均线
        ax1.plot(df_features['date'], df_features['close'], label='价格', alpha=0.7)
        ax1.plot(df_features['date'], df_features['SMA_20'], label='SMA20', alpha=0.8)
        ax1.plot(df_features['date'], df_features['SMA_50'], label='SMA50', alpha=0.8)
        ax1.set_title('价格走势和移动平均线')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True)
        
        # 投资组合价值
        ax2.plot(df_features['date'], portfolio_values, label='投资组合价值', color='green')
        ax2.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label='初始资金')
        ax2.set_title('投资组合价值变化')
        ax2.set_ylabel('价值')
        ax2.set_xlabel('日期')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/demo_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("演示完成！结果图表已保存到 results/demo_results.png")
        
        return True
        
    except Exception as e:
        logger.error(f"演示运行失败: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("中国原油期货量化交易系统演示")
    logger.info("="*50)
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 运行演示
    success = run_demo()
    
    if success:
        logger.info("\n🎉 演示成功完成！")
        logger.info("您可以查看 results/demo_results.png 查看结果图表")
        logger.info("要运行完整系统，请使用: python main.py --mode full")
    else:
        logger.error("\n❌ 演示运行失败")
        logger.error("请检查系统配置和依赖项")

if __name__ == "__main__":
    main() 