import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from typing import Dict, List, Tuple, Optional
import os
import yaml
import logging
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import re

from src.models.signal_weight_env import SignalWeightTradingEnv
from src.models.max_weight_env import MaxWeightTradingEnv
logger = logging.getLogger(__name__)

class RLTrainer:
    """强化学习模型训练器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model_training']['model']
        self.training_config = self.config['model_training']['training']
        self.storage_config = self.config['model_training']['storage']
        
        # 确保模型存储目录存在
        os.makedirs(self.storage_config['model_path'], exist_ok=True)
        os.makedirs(self.storage_config['results_path'], exist_ok=True)
        
        # 设置随机种子
        set_random_seed(self.training_config['random_state'])
        
        # 初始化模型
        self.model = None
        self.env = None
        
    def create_env(self, df: pd.DataFrame, env_type: str = None, config: dict = None) -> object:
        """
        只允许通过config['model_training']['model']['env_module']和env_name指定环境文件和类。
        env_module: 环境文件名（不带.py），如 'signal_weight_env' 或 'max_weight_env'
        env_name: 环境类名，如 'SignalWeightTradingEnv' 或 'MaxWeightTradingEnv'
        """
        if config is None:
            raise ValueError("必须传递config参数，且在config['model_training']['model']中指定env_module和env_name")
        env_class = config['model_training']['model'].get('env_name', None)
        env_module = config['model_training']['model'].get('env_module', None)
        if not env_class or not env_module:
            raise ValueError("config['model_training']['model']中必须指定env_module和env_name")
        module = importlib.import_module(f"src.models.{env_module}")
        env_cls = getattr(module, env_class)
        # 自动读取环境参数
        env_param_key = env_module  # 'signal_weight_env' 或 'max_weight_env'
        env_kwargs = config['model_training'].get(env_param_key, {})
        return env_cls(df, "config/config.yaml", env_kwargs=env_kwargs)
    
    def create_model(self, env: object, algorithm: str = None) -> object:
        """创建强化学习模型
        
        Args:
            env: 交易环境
            algorithm: 算法名称
            
        Returns:
            强化学习模型
        """
        algorithm = algorithm or self.model_config['algorithm']
        algo = self.model_config['algorithm']
        env_name = self.model_config['env_name']
        # 包装环境
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        # 创建模型
        if algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.model_config['learning_rate'],
                batch_size=self.model_config['batch_size'],
                gamma=self.model_config['gamma'],
                gae_lambda=self.model_config['gae_lambda'],
                clip_range=self.model_config['clip_range'],
                ent_coef=self.model_config['ent_coef'],
                vf_coef=self.model_config['vf_coef'],
                max_grad_norm=self.model_config['max_grad_norm'],
                verbose=1,
                tensorboard_log=f"logs/{algo}_{env_name}/tensorboard/"
            )
        elif algorithm == "A2C":
            self.model = A2C(
                "MlpPolicy",
                env,
                learning_rate=self.model_config['learning_rate'],
                gamma=self.model_config['gamma'],
                gae_lambda=self.model_config['gae_lambda'],
                ent_coef=self.model_config['ent_coef'],
                vf_coef=self.model_config['vf_coef'],
                max_grad_norm=self.model_config['max_grad_norm'],
                verbose=1,
                tensorboard_log=f"logs/{algo}_{env_name}/tensorboard/"
            )
        elif algorithm == "DQN":
            self.model = DQN(
                "MlpPolicy",
                env,
                learning_rate=self.model_config['learning_rate'],
                gamma=self.model_config['gamma'],
                verbose=1,
                tensorboard_log=f"logs/{algo}_{env_name}/tensorboard/"
            )
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        return self.model
    
    def train_model(self, df: pd.DataFrame, env_type: str = None, 
                   algorithm: str = None, total_timesteps: int = None, config: dict = None) -> Dict:
        """训练模型"""
        total_timesteps = total_timesteps or self.model_config['total_timesteps']
        train_df, val_df, test_df = self._split_data(df)
        env = self.create_env(train_df, env_type, config)
        model = self.create_model(env, algorithm)
        callbacks = self._setup_callbacks(env, val_df, env_type, config)
        logger.info(f"开始训练 {algorithm} 模型，总步数: {total_timesteps}")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        algo = self.model_config['algorithm']
        env_name = self.model_config['env_name']
        model_path = f"{self.storage_config['model_path']}/{algo}_{env_name}_model"
        model.save(model_path)
        logger.info(f"模型已保存到: {model_path}")
        evaluation_results = self.evaluate_model(model, test_df, env_type, config)
        # 保存训练结果
        results_dir = f"{self.storage_config['model_path']}/training_results/"
        os.makedirs(results_dir, exist_ok=True)
        # 保存训练配置
        portfolio_stats = evaluation_results.get('portfolio_stats', {})
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
            'env_name': env_name,
            'total_timesteps': total_timesteps,
            'model_path': model_path,
            'portfolio_stats': portfolio_stats_serializable
        }
        import json
        with open(f"{results_dir}/{algo}_{env_name}_training_config.json", 'w', encoding='utf-8') as f:
            json.dump(training_config, f, ensure_ascii=False, indent=2)
        logger.info(f"训练配置已保存到: {results_dir}/{algo}_{env_name}_training_config.json")
        # 保存评估结果
        eval_file = f"{results_dir}/{algo}_{env_name}_evaluation_results.json"
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
            json.dump(to_serializable(evaluation_results), f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {eval_file}")
        return {
            'model': model,
            'model_path': model_path,
            'evaluation_results': evaluation_results,
            'training_config': {
                'algorithm': algorithm,
                'env_type': env_type,
                'total_timesteps': total_timesteps
            }
        }
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """分割数据为训练集、验证集和测试集"""
        train_split = self.training_config['train_split']
        validation_split = self.training_config['validation_split']
        test_split = self.training_config['test_split']
        n = len(df)
        train_end = int(n * train_split)
        val_end = train_end + int(n * validation_split)
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        logger.info(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}, 测试集大小: {len(test_df)}")
        return train_df, val_df, test_df
    
    def _setup_callbacks(self, env: DummyVecEnv, test_df: pd.DataFrame, env_type: str = None, config: dict = None) -> List:
        """设置回调函数"""
        callbacks = []
        algo = self.model_config['algorithm']
        env_name = self.model_config['env_name']
        # checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{self.storage_config['model_path']}/checkpoints/",
            name_prefix=f"{algo}_{env_name}_rl_model"
        )
        callbacks.append(checkpoint_callback)
        # 评估回调（使用相同的环境类型）
        eval_env = self.create_env(test_df, env_type, config)
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.storage_config['model_path']}/best/{algo}_{env_name}_best_model/",
            log_path=f"logs/{algo}_{env_name}/eval/",
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        return callbacks
    
    def evaluate_model(self, model: object, test_df: pd.DataFrame, env_type: str = None, config: dict = None) -> Dict:
        """评估模型"""
        # 创建测试环境
        test_env = self.create_env(test_df, env_type, config)
        # 运行测试
        obs = test_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = test_env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
            total_reward += reward
            step_count += 1
        portfolio_stats = test_env.get_portfolio_stats()
        evaluation_results = {
            'total_reward': total_reward,
            'step_count': step_count,
            'portfolio_stats': portfolio_stats,
            'trades': test_env.trades
        }
        logger.info(f"模型评估完成，总奖励: {total_reward:.4f}")
        logger.info(f"投资组合统计: {portfolio_stats}")
        return evaluation_results
    
    def load_model(self, model_path: str, algorithm: str = None) -> object:
        """加载已训练的模型
        
        Args:
            model_path: 模型路径
            algorithm: 算法名称
            
        Returns:
            加载的模型
        """
        algorithm = algorithm or self.model_config['algorithm']
        
        if algorithm == "PPO":
            self.model = PPO.load(model_path)
        elif algorithm == "A2C":
            self.model = A2C.load(model_path)
        elif algorithm == "DQN":
            self.model = DQN.load(model_path)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        logger.info(f"模型已从 {model_path} 加载")
        
        return self.model
    
    def plot_training_results(self, results: Dict, save_path: str = None):
        """绘制训练结果
        
        Args:
            results: 训练结果
            save_path: 保存路径
        """
        if save_path is None:
            save_path = f"{self.storage_config['results_path']}/training_results.png"
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 策略比较
        strategies = list(results['results'].keys())
        valid_results = {k: v for k, v in results['results'].items() if v is not None}
        
        if valid_results:
            # 总收益率比较
            returns = [valid_results[s]['portfolio_stats']['total_return'] for s in valid_results.keys()]
            axes[0, 0].bar(valid_results.keys(), returns)
            axes[0, 0].set_title('总收益率比较')
            axes[0, 0].set_ylabel('总收益率')
            
            # 夏普比率比较
            sharpe_ratios = [valid_results[s]['portfolio_stats']['sharpe_ratio'] for s in valid_results.keys()]
            axes[0, 1].bar(valid_results.keys(), sharpe_ratios)
            axes[0, 1].set_title('夏普比率比较')
            axes[0, 1].set_ylabel('夏普比率')
            
            # 最大回撤比较
            max_drawdowns = [valid_results[s]['portfolio_stats']['max_drawdown'] for s in valid_results.keys()]
            axes[1, 0].bar(valid_results.keys(), max_drawdowns)
            axes[1, 0].set_title('最大回撤比较')
            axes[1, 0].set_ylabel('最大回撤')
            
            # 交易次数比较
            trade_counts = [valid_results[s]['portfolio_stats']['total_trades'] for s in valid_results.keys()]
            axes[1, 1].bar(valid_results.keys(), trade_counts)
            axes[1, 1].set_title('交易次数比较')
            axes[1, 1].set_ylabel('交易次数')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"训练结果图表已保存到: {save_path}")

if __name__ == "__main__":
    # 测试强化学习训练器
    from src.data.data_loader import OilDataLoader
    from src.features.feature_engineering import FeatureEngineer
    
    # 加载数据
    loader = OilDataLoader()
    df = loader.get_data_with_cache(use_cache=True)
    
    # 特征工程
    engineer = FeatureEngineer()
    df_features = engineer.add_technical_indicators(df)
    df_features = engineer.add_target_variables(df_features)
    df_features = engineer.create_trading_features(df_features)
    
    # 创建训练器
    trainer = RLTrainer()
    
    # 训练模型
    results = trainer.train_model(df_features, env_type="standard", total_timesteps=50000)
    
    print("模型训练完成!")
    print(f"模型路径: {results['model_path']}")
    print(f"评估结果: {results['evaluation_results']['portfolio_stats']}") 