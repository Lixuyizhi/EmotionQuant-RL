#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器
负责处理配置文件的加载、继承和合并
"""

import yaml
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.raw_config = {}
        self.processed_config = {}
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件
        
        Returns:
            处理后的配置字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.raw_config = yaml.safe_load(f)
            
            # 处理配置继承
            self.processed_config = self._process_inheritance(self.raw_config)
            logger.info("配置文件加载成功")
            return self.processed_config
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def _process_inheritance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理配置继承
        
        Args:
            config: 原始配置字典
            
        Returns:
            处理继承后的配置字典
        """
        processed_config = config.copy()
        
        # 处理交易环境配置继承
        if 'trading_env' in processed_config:
            trading_env = processed_config['trading_env']
            
            # 处理环境特定的继承
            for env_name, env_config in trading_env.items():
                if env_name != 'common' and isinstance(env_config, dict):
                    inherit_from = env_config.get('inherit_from')
                    if inherit_from and inherit_from in trading_env:
                        # 合并通用配置和环境特定配置
                        base_config = trading_env[inherit_from].copy()
                        # 环境特定配置覆盖通用配置
                        base_config.update(env_config)
                        # 移除继承标记
                        base_config.pop('inherit_from', None)
                        trading_env[env_name] = base_config
        
        return processed_config
    
    def get_env_config(self, env_name: str) -> Dict[str, Any]:
        """获取指定环境的配置
        
        Args:
            env_name: 环境名称 (如 'signal_weight_env', 'max_weight_env')
            
        Returns:
            环境配置字典
        """
        if 'trading_env' not in self.processed_config:
            raise ValueError("配置文件中缺少 trading_env 部分")
        
        if env_name not in self.processed_config['trading_env']:
            raise ValueError(f"环境 {env_name} 不存在")
        
        return self.processed_config['trading_env'][env_name]
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型训练配置
        
        Returns:
            模型训练配置字典
        """
        return self.processed_config.get('model_training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据处理配置
        
        Returns:
            数据处理配置字典
        """
        return self.processed_config.get('data_processing', {})
    
    def get_backtest_config(self) -> Dict[str, Any]:
        """获取回测配置
        
        Returns:
            回测配置字典
        """
        return self.processed_config.get('backtest', {})
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置
        
        Args:
            updates: 要更新的配置字典
        """
        # 深度更新配置
        self._deep_update(self.processed_config, updates)
        logger.info("配置已更新")
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """深度更新字典
        
        Args:
            base_dict: 基础字典
            update_dict: 更新字典
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """保存配置到文件
        
        Args:
            output_path: 输出文件路径，如果为None则覆盖原文件
        """
        if output_path is None:
            output_path = self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.processed_config, f, default_flow_style=False, 
                          allow_unicode=True, sort_keys=False)
            logger.info(f"配置已保存到: {output_path}")
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            raise
    
    def validate_config(self) -> bool:
        """验证配置的有效性
        
        Returns:
            配置是否有效
        """
        try:
            # 检查必要的配置项
            required_sections = ['data_processing', 'model_training', 'trading_env']
            for section in required_sections:
                if section not in self.processed_config:
                    logger.error(f"缺少必要的配置部分: {section}")
                    return False
            
            # 检查交易环境配置
            trading_env = self.processed_config['trading_env']
            if 'common' not in trading_env:
                logger.error("缺少通用交易环境配置")
                return False
            
            # 检查环境特定配置
            env_names = ['signal_weight_env', 'max_weight_env']
            for env_name in env_names:
                if env_name in trading_env:
                    env_config = trading_env[env_name]
                    required_params = ['position_size', 'buy_threshold', 'sell_threshold']
                    for param in required_params:
                        if param not in env_config:
                            logger.error(f"环境 {env_name} 缺少必要参数: {param}")
                            return False
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def get_config_summary(self) -> str:
        """获取配置摘要
        
        Returns:
            配置摘要字符串
        """
        summary = []
        summary.append("=" * 50)
        summary.append("配置摘要")
        summary.append("=" * 50)
        
        # 数据处理配置
        data_config = self.get_data_config()
        summary.append(f"数据源: {data_config.get('data_source', {}).get('source', 'N/A')}")
        summary.append(f"交易品种: {data_config.get('data_source', {}).get('symbol', 'N/A')}")
        summary.append(f"技术指标: {', '.join(data_config.get('features', {}).get('technical_indicators', []))}")
        
        # 模型训练配置
        model_config = self.get_model_config()
        summary.append(f"算法: {model_config.get('model', {}).get('algorithm', 'N/A')}")
        summary.append(f"环境: {model_config.get('model', {}).get('env_name', 'N/A')}")
        summary.append(f"训练步数: {model_config.get('model', {}).get('total_timesteps', 'N/A')}")
        
        # 交易环境配置
        trading_env = self.processed_config.get('trading_env', {})
        summary.append(f"启用的环境: {', '.join([k for k in trading_env.keys() if k != 'common'])}")
        
        summary.append("=" * 50)
        return "\n".join(summary)
