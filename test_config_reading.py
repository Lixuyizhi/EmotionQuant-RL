#!/usr/bin/env python3
"""
配置读取演示脚本
展示如何从config.yaml读取信号权重环境参数
"""

import yaml

def load_config():
    """加载配置文件"""
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def demo_config_reading():
    """演示配置读取过程"""
    print("=== 配置读取演示 ===\n")
    
    # 1. 加载整个配置文件
    config = load_config()
    print("1. 加载的完整配置结构:")
    print(f"   config.keys() = {list(config.keys())}")
    print()
    
    # 2. 获取model_training部分
    model_training = config.get('model_training', {})
    print("2. 获取model_training部分:")
    print(f"   model_training.keys() = {list(model_training.keys())}")
    print()
    
    # 3. 获取signal_weight_env部分
    env_config = model_training.get('signal_weight_env', {})
    print("3. 获取signal_weight_env部分:")
    print(f"   env_config.keys() = {list(env_config.keys())}")
    print()
    
    # 4. 读取具体参数
    print("4. 读取的具体参数值:")
    print(f"   初始余额: {env_config.get('initial_balance', 100000)}")
    print(f"   手续费率: {env_config.get('transaction_fee', 0.001):.4f}")
    print(f"   滑点率: {env_config.get('slippage', 0.0005):.4f}")
    print(f"   仓位大小: {env_config.get('position_size', 0.1):.2f}")
    print(f"   买入阈值: {env_config.get('buy_threshold', 0.1)}")
    print(f"   卖出阈值: {env_config.get('sell_threshold', -0.1)}")
    print(f"   最大仓位比例: {env_config.get('max_position_ratio', 0.8):.2f}")
    print(f"   最小交易金额: {env_config.get('min_trade_amount', 1000)}")
    print(f"   奖励缩放因子: {env_config.get('reward_scale', 1.0):.2f}")
    print(f"   风险惩罚系数: {env_config.get('risk_penalty', 0.01):.3f}")
    print()
    
    # 5. 展示配置路径
    print("5. 配置读取路径:")
    print("   config.yaml")
    print("   └── model_training")
    print("       └── signal_weight_env")
    print("           ├── initial_balance: 100000")
    print("           ├── transaction_fee: 0.001")
    print("           ├── slippage: 0.0005")
    print("           └── ... (其他参数)")
    print()
    
    # 6. 代码中的读取方式
    print("6. 代码中的读取方式:")
    print("   env_config = self.config.get('model_training', {}).get('signal_weight_env', {})")
    print("   self.initial_balance = env_config.get('initial_balance', 100000)")
    print("   self.transaction_fee = env_config.get('transaction_fee', 0.001)")
    print("   # ... 其他参数")

if __name__ == '__main__':
    demo_config_reading() 