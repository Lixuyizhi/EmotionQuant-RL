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

from src.models.trading_env import OilTradingEnv, MultiFactorTradingEnv, MaxWeightTradingEnv
from src.models.signal_weight_env import SignalWeightTradingEnv

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
        
        # 修正：适配新配置结构
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
        
    def create_env(self, df: pd.DataFrame, env_type: str = "standard") -> OilTradingEnv:
        """创建交易环境
        
        Args:
            df: 数据DataFrame
            env_type: 环境类型 ("standard", "weighted", "max_weight", "signal_weight")
            
        Returns:
            交易环境实例
        """
        if env_type == "standard":
            self.env = OilTradingEnv(df, "config/config.yaml")
        elif env_type == "weighted":
            # 使用简单的等权重
            feature_columns = [col for col in df.columns 
                              if col not in ['date', 'open', 'high', 'low', 'close', 'volume'] 
                              and not col.startswith('target_')]
            feature_weights = {feature: 1.0/len(feature_columns) for feature in feature_columns}
            self.env = MultiFactorTradingEnv(df, feature_weights, "config/config.yaml")
        elif env_type == "max_weight":
            # 使用简单的等权重
            feature_columns = [col for col in df.columns 
                              if col not in ['date', 'open', 'high', 'low', 'close', 'volume'] 
                              and not col.startswith('target_')]
            feature_weights = {feature: 1.0/len(feature_columns) for feature in feature_columns}
            self.env = MaxWeightTradingEnv(df, feature_weights, "config/config.yaml")
        elif env_type == "signal_weight":
            # 信号权重环境：学习给RSI_signal、BB_signal、SMA_signal分配权重
            self.env = SignalWeightTradingEnv(df, "config/config.yaml")
        else:
            raise ValueError(f"不支持的环境类型: {env_type}")
        
        return self.env
    
    def create_model(self, env: OilTradingEnv, algorithm: str = None) -> object:
        """创建强化学习模型
        
        Args:
            env: 交易环境
            algorithm: 算法名称
            
        Returns:
            强化学习模型
        """
        algorithm = algorithm or self.model_config['algorithm']
        
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
                tensorboard_log=f"{self.storage_config['log_path']}/tensorboard/"
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
                tensorboard_log=f"{self.storage_config['log_path']}/tensorboard/"
            )
        elif algorithm == "DQN":
            self.model = DQN(
                "MlpPolicy",
                env,
                learning_rate=self.model_config['learning_rate'],
                gamma=self.model_config['gamma'],
                verbose=1,
                tensorboard_log=f"{self.storage_config['log_path']}/tensorboard/"
            )
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        return self.model
    
    def train_model(self, df: pd.DataFrame, env_type: str = "standard", 
                   algorithm: str = None, total_timesteps: int = None) -> Dict:
        """训练模型
        
        Args:
            df: 数据DataFrame
            env_type: 环境类型
            algorithm: 算法名称
            total_timesteps: 训练步数
            
        Returns:
            训练结果字典
        """
        total_timesteps = total_timesteps or self.model_config['total_timesteps']
        
        # 数据分割
        train_df, test_df = self._split_data(df)
        
        # 创建环境
        env = self.create_env(train_df, env_type)
        
        # 创建模型
        model = self.create_model(env, algorithm)
        
        # 设置回调函数
        callbacks = self._setup_callbacks(env, test_df, env_type)
        
        logger.info(f"开始训练 {algorithm} 模型，总步数: {total_timesteps}")
        
        # 训练模型
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # 保存模型
        model_path = f"{self.storage_config['model_path']}/{algorithm}_{env_type}_model"
        model.save(model_path)
        logger.info(f"模型已保存到: {model_path}")
        
        # 评估模型
        evaluation_results = self.evaluate_model(model, test_df, env_type)
        
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
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分割数据为训练集和测试集
        
        Args:
            df: 数据DataFrame
            
        Returns:
            训练集和测试集
        """
        train_split = self.training_config['train_split']
        test_split = self.training_config['test_split']
        
        # 按时间分割
        split_idx = int(len(df) * train_split)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
        
        return train_df, test_df
    
    def _setup_callbacks(self, env: DummyVecEnv, test_df: pd.DataFrame, env_type: str = "standard") -> List:
        """设置回调函数
        
        Args:
            env: 训练环境
            test_df: 测试数据
            env_type: 环境类型
            
        Returns:
            回调函数列表
        """
        callbacks = []
        
        # 检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{self.storage_config['model_path']}/checkpoints/",
            name_prefix="rl_model"
        )
        callbacks.append(checkpoint_callback)
        
        # 评估回调（使用相同的环境类型）
        eval_env = self.create_env(test_df, env_type)
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env])
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.storage_config['model_path']}/best/",
            log_path=f"{self.storage_config['results_path']}/logs/",
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        return callbacks
    
    def evaluate_model(self, model: object, test_df: pd.DataFrame, 
                      env_type: str = "standard") -> Dict:
        """评估模型
        
        Args:
            model: 训练好的模型
            test_df: 测试数据
            env_type: 环境类型
            
        Returns:
            评估结果
        """
        # 创建测试环境
        test_env = self.create_env(test_df, env_type)
        
        # 运行测试
        obs = test_env.reset()
        # 处理Gym API vs SB3 API兼容性
        if isinstance(obs, tuple):
            obs = obs[0]  # 如果是元组，取第一个元素（观察）
        
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = test_env.step(action)
            
            # 处理Gym API vs SB3 API兼容性
            if len(step_result) == 4:  # Gym API: (obs, reward, done, info)
                obs, reward, done, info = step_result
            else:  # SB3 API: (obs, reward, done, truncated, info)
                obs, reward, done, truncated, info = step_result
                done = done or truncated
            
            total_reward += reward
            step_count += 1
        
        # 获取投资组合统计
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
    
    def compare_strategies(self, df: pd.DataFrame) -> Dict:
        """比较不同策略的性能
        
        Args:
            df: 数据DataFrame
            
        Returns:
            策略比较结果
        """
        strategies = ["standard", "weighted", "max_weight"]
        results = {}
        
        for strategy in strategies:
            logger.info(f"训练 {strategy} 策略...")
            
            try:
                # 训练模型
                train_result = self.train_model(df, env_type=strategy)
                
                # 保存结果
                results[strategy] = {
                    'portfolio_stats': train_result['evaluation_results']['portfolio_stats'],
                    'total_reward': train_result['evaluation_results']['total_reward'],
                    'model_path': train_result['model_path']
                }
                
            except Exception as e:
                logger.error(f"训练 {strategy} 策略失败: {str(e)}")
                results[strategy] = None
        
        # 生成比较报告
        comparison_report = self._generate_comparison_report(results)
        
        return {
            'results': results,
            'comparison_report': comparison_report
        }
    
    def _generate_comparison_report(self, results: Dict) -> Dict:
        """生成策略比较报告
        
        Args:
            results: 策略结果字典
            
        Returns:
            比较报告
        """
        report = {
            'summary': {},
            'best_strategy': None,
            'recommendations': []
        }
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return report
        
        # 找出最佳策略
        best_strategy = max(valid_results.items(), 
                          key=lambda x: x[1]['portfolio_stats']['sharpe_ratio'])
        
        report['best_strategy'] = best_strategy[0]
        
        # 生成摘要
        for strategy, result in valid_results.items():
            stats = result['portfolio_stats']
            report['summary'][strategy] = {
                'total_return': f"{stats['total_return']:.4f}",
                'sharpe_ratio': f"{stats['sharpe_ratio']:.4f}",
                'max_drawdown': f"{stats['max_drawdown']:.4f}",
                'volatility': f"{stats['volatility']:.4f}",
                'total_trades': stats['total_trades']
            }
        
        # 生成建议
        if best_strategy[1]['portfolio_stats']['sharpe_ratio'] > 1.0:
            report['recommendations'].append("策略表现良好，夏普比率超过1.0")
        
        if best_strategy[1]['portfolio_stats']['max_drawdown'] < 0.2:
            report['recommendations'].append("最大回撤控制在20%以内，风险控制良好")
        
        return report
    
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