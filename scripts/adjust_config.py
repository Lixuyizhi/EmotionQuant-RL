#!/usr/bin/env python3
"""
配置调整脚本
直接修改 config/config.yaml 文件中的信号权重环境参数
"""

import yaml
import argparse
import os
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config: dict, config_path: str = "config/config.yaml"):
    """保存配置文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

def print_current_config():
    """打印当前配置"""
    config = load_config()
    env_config = config.get('model_training', {}).get('signal_weight_env', {})
    
    print("=== 当前信号权重环境配置 ===")
    print(f"初始余额: {env_config.get('initial_balance', 100000)}")
    print(f"手续费率: {env_config.get('transaction_fee', 0.001):.4f}")
    print(f"滑点率: {env_config.get('slippage', 0.0005):.4f}")
    print(f"仓位大小: {env_config.get('position_size', 0.1):.2f}")
    print(f"买入阈值: {env_config.get('buy_threshold', 0.1)}")
    print(f"卖出阈值: {env_config.get('sell_threshold', -0.1)}")
    print(f"最大仓位比例: {env_config.get('max_position_ratio', 0.8):.2f}")
    print(f"最小交易金额: {env_config.get('min_trade_amount', 1000)}")
    print(f"奖励缩放因子: {env_config.get('reward_scale', 1.0):.2f}")
    print(f"风险惩罚系数: {env_config.get('risk_penalty', 0.01):.3f}")
    print("============================")

def apply_preset_config(preset: str):
    """应用预设配置"""
    presets = {
        'conservative': {
            'transaction_fee': 0.002,
            'slippage': 0.001,
            'position_size': 0.05,
            'max_position_ratio': 0.6,
            'buy_threshold': 0.2,
            'sell_threshold': -0.2,
            'risk_penalty': 0.02
        },
        'aggressive': {
            'transaction_fee': 0.0005,
            'slippage': 0.0002,
            'position_size': 0.2,
            'max_position_ratio': 0.9,
            'buy_threshold': 0.05,
            'sell_threshold': -0.05,
            'risk_penalty': 0.005
        },
        'high_frequency': {
            'position_size': 0.3,
            'min_trade_amount': 500,
            'buy_threshold': 0.02,
            'sell_threshold': -0.02,
            'reward_scale': 0.5
        },
        'long_term': {
            'position_size': 0.05,
            'min_trade_amount': 5000,
            'buy_threshold': 0.3,
            'sell_threshold': -0.3,
            'risk_penalty': 0.005
        }
    }
    
    if preset not in presets:
        raise ValueError(f"不支持的预设配置: {preset}。可用预设: {list(presets.keys())}")
    
    config = load_config()
    
    # 确保配置结构存在
    if 'model_training' not in config:
        config['model_training'] = {}
    if 'signal_weight_env' not in config['model_training']:
        config['model_training']['signal_weight_env'] = {}
    
    # 应用预设参数
    config['model_training']['signal_weight_env'].update(presets[preset])
    
    save_config(config)
    print(f"已应用 {preset} 预设配置")

def update_config(**kwargs):
    """更新配置参数"""
    config = load_config()
    
    # 确保配置结构存在
    if 'model_training' not in config:
        config['model_training'] = {}
    if 'signal_weight_env' not in config['model_training']:
        config['model_training']['signal_weight_env'] = {}
    
    # 更新参数
    for key, value in kwargs.items():
        if value is not None:
            config['model_training']['signal_weight_env'][key] = value
    
    save_config(config)
    print("已更新配置参数")

def main():
    parser = argparse.ArgumentParser(description='调整信号权重环境配置')
    parser.add_argument('--preset', choices=['conservative', 'aggressive', 'high_frequency', 'long_term'], 
                       help='应用预设配置')
    parser.add_argument('--initial_balance', type=float, help='初始余额')
    parser.add_argument('--transaction_fee', type=float, help='手续费率')
    parser.add_argument('--slippage', type=float, help='滑点率')
    parser.add_argument('--position_size', type=float, help='仓位大小')
    parser.add_argument('--buy_threshold', type=float, help='买入阈值')
    parser.add_argument('--sell_threshold', type=float, help='卖出阈值')
    parser.add_argument('--max_position_ratio', type=float, help='最大仓位比例')
    parser.add_argument('--min_trade_amount', type=float, help='最小交易金额')
    parser.add_argument('--reward_scale', type=float, help='奖励缩放因子')
    parser.add_argument('--risk_penalty', type=float, help='风险惩罚系数')
    parser.add_argument('--show', action='store_true', help='显示当前配置')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists("config/config.yaml"):
        print("错误: 配置文件 config/config.yaml 不存在")
        return
    
    # 显示当前配置
    if args.show:
        print_current_config()
        return
    
    # 应用预设配置
    if args.preset:
        apply_preset_config(args.preset)
    
    # 应用自定义参数
    custom_params = {
        'initial_balance': args.initial_balance,
        'transaction_fee': args.transaction_fee,
        'slippage': args.slippage,
        'position_size': args.position_size,
        'buy_threshold': args.buy_threshold,
        'sell_threshold': args.sell_threshold,
        'max_position_ratio': args.max_position_ratio,
        'min_trade_amount': args.min_trade_amount,
        'reward_scale': args.reward_scale,
        'risk_penalty': args.risk_penalty
    }
    
    # 过滤掉None值
    custom_params = {k: v for k, v in custom_params.items() if v is not None}
    
    if custom_params:
        update_config(**custom_params)
        print("已应用自定义参数:")
        for key, value in custom_params.items():
            print(f"  {key}: {value}")
    
    # 显示更新后的配置
    print("\n更新后的配置:")
    print_current_config()

if __name__ == '__main__':
    main() 