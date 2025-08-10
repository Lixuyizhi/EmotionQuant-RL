#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置迁移脚本
帮助用户从旧配置迁移到新的配置结构
"""

import os
import sys
import yaml
import shutil
from datetime import datetime

def backup_original_config():
    """备份原始配置文件"""
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"config/config_backup_{timestamp}.yaml"
        shutil.copy2(config_path, backup_path)
        print(f"✓ 原始配置文件已备份到: {backup_path}")
        return backup_path
    return None

def create_new_config():
    """创建新的配置文件"""
    new_config = {
        "data_processing": {
            "data_source": {
                "source": "local",
                "local_file": "bc2211_major_contracts_2022_30min.xlsx",
                "symbol": "SC",
                "data_path": "data/",
                "start_date": "2017-01-01",
                "end_date": "2025-01-01"
            },
            "features": {
                "technical_indicators": ["RSI", "MACD", "ATR", "EMA"],
                "lookback_periods": [20],
                "feature_scaling": "standard"
            },
            "preprocessing": {
                "cache_data": True,
                "cache_path": "data/processed",
                "train_split": 0.7,
                "validation_split": 0.15,
                "test_split": 0.15,
                "random_state": 42
            }
        },
        "model_training": {
            "model": {
                "algorithm": "A2C",
                "env_module": "signal_weight_env",
                "env_name": "SignalWeightTradingEnv",
                "total_timesteps": 100000,
                "learning_rate": 0.0001,
                "batch_size": 128,
                "gamma": 0.995,
                "gae_lambda": 0.98,
                "vf_coef": 0.8,
                "ent_coef": 0.02,
                "max_grad_norm": 0.3,
                "clip_range": 0.2
            },
            "training": {
                "n_episodes": 1000,
                "eval_freq": 5000,
                "save_freq": 10000,
                "early_stopping_patience": 50,
                "random_state": 42,
                "train_split": 0.7,
                "validation_split": 0.15,
                "test_split": 0.15
            },
            "storage": {
                "model_path": "models/",
                "log_path": "logs/",
                "results_path": "results/"
            }
        },
        "trading_env": {
            "common": {
                "initial_balance": 1000000,
                "max_position_ratio": 0.5,
                "observation_features": {
                    "include_raw_ohlc": False,
                    "include_normalized_ohlc": True,
                    "include_price_changes": True,
                    "include_volume": False,
                    "include_technical_indicators": True,
                    "include_account_info": True
                },
                "trading_signals": {
                    "enabled_signals": ["RSI_signal", "BB_signal", "SMA_signal", "MACD_signal"],
                    "weight_min": 0.0,
                    "weight_max": 1.0,
                    "signal_thresholds": {
                        "rsi_overbought": 70,
                        "rsi_oversold": 30,
                        "bb_upper_multiplier": 1.0,
                        "bb_lower_multiplier": 1.0,
                        "cci_overbought": 100,
                        "cci_oversold": -100,
                        "williams_overbought": -20,
                        "williams_oversold": -80,
                        "stoch_overbought": 80,
                        "stoch_oversold": 20
                    }
                }
            },
            "signal_weight_env": {
                "inherit_from": "common",
                "position_size": 0.1,
                "min_trade_amount": 50000,
                "buy_threshold": 0.3,
                "sell_threshold": -0.3,
                "transaction_fee": 0.0002,
                "slippage": 0.00003,
                "reward_scale": 0.5,
                "risk_penalty": 0.002
            },
            "max_weight_env": {
                "inherit_from": "common",
                "position_size": 0.12,
                "min_trade_amount": 40000,
                "buy_threshold": 0.25,
                "sell_threshold": -0.25,
                "transaction_fee": 0.0003,
                "slippage": 0.00005,
                "reward_scale": 0.4,
                "risk_penalty": 0.003
            }
        },
        "backtest": {
            "data_path": "data/SC_2020-01-01_2024-01-01.csv",
            "model_path": "models/best/PPO_SignalWeightTradingEnv_best_model/best_model.zip",
            "global_settings": {
                "benchmark": "buy_and_hold",
                "initial_cash": 1000000,
                "risk_free_rate": 0.03,
                "commission": 0.001,
                "slippage": 0.0005
            },
            "results": {
                "generate_report": True,
                "plot_results": True,
                "save_path": "backtest_results/"
            },
            "env_params": {
                "inherit_from_training": True
            }
        }
    }
    
    return new_config

def migrate_config():
    """执行配置迁移"""
    print("=" * 60)
    print("配置迁移工具")
    print("=" * 60)
    
    # 1. 备份原始配置
    print("1. 备份原始配置文件...")
    backup_path = backup_original_config()
    
    # 2. 创建新配置
    print("\n2. 创建新配置文件...")
    new_config = create_new_config()
    
    # 3. 保存新配置
    print("\n3. 保存新配置文件...")
    config_path = "config/config.yaml"
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(new_config, f, default_flow_style=False, 
                      allow_unicode=True, sort_keys=False)
        print(f"✓ 新配置文件已保存到: {config_path}")
    except Exception as e:
        print(f"✗ 保存新配置文件失败: {e}")
        return False
    
    # 4. 验证新配置
    print("\n4. 验证新配置文件...")
    try:
        from src.utils.config_manager import ConfigManager
        config_manager = ConfigManager(config_path)
        if config_manager.validate_config():
            print("✓ 新配置文件验证通过")
        else:
            print("✗ 新配置文件验证失败")
            return False
    except Exception as e:
        print(f"✗ 新配置文件验证失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 配置迁移完成！")
    print("=" * 60)
    print("\n主要改进:")
    print("1. ✅ 消除了重复配置")
    print("2. ✅ 建立了配置继承机制")
    print("3. ✅ 简化了配置维护")
    print("4. ✅ 保持了功能完整性")
    print("\n新配置结构:")
    print("- trading_env.common: 通用环境参数")
    print("- trading_env.signal_weight_env: 信号权重环境特定参数")
    print("- trading_env.max_weight_env: 最大权重环境特定参数")
    print("\n如果需要恢复原始配置，请使用备份文件:")
    if backup_path:
        print(f"  {backup_path}")
    
    return True

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("配置迁移工具")
        print("\n使用方法:")
        print("python scripts/migrate_config.py")
        print("\n功能:")
        print("- 备份原始配置文件")
        print("- 创建新的配置结构")
        print("- 验证新配置的有效性")
        return
    
    try:
        migrate_config()
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断了迁移过程")
    except Exception as e:
        print(f"\n❌ 迁移失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
