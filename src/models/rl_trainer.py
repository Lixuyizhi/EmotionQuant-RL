import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
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
        final_model_dir = f"{self.storage_config['model_path']}/Final Model"
        os.makedirs(final_model_dir, exist_ok=True)
        model_path = f"{final_model_dir}/{algo}_{env_name}_model"
        model.save(model_path)
        logger.info(f"模型已保存到: {model_path}")
        
        # 获取训练过程中的权重历史记录
        weights_history = []
        signal_names = []
        if hasattr(env.envs[0], 'signal_weights_history') and env.envs[0].signal_weights_history:
            weights_history = env.envs[0].signal_weights_history
            signal_names = env.envs[0].enabled_signals if hasattr(env.envs[0], 'enabled_signals') else []
            logger.info(f"获取到权重历史记录，共 {len(weights_history)} 个时间步")
        
        # 记录最终模型在每个交易时段的权重分配
        final_weights_data = self._record_final_model_weights(model, env, train_df, signal_names, final_model_dir, algo, env_name)
        
        evaluation_results = self.evaluate_model(model, test_df, env_type, config)
        
        # 获取权重分析（如果环境支持）
        weights_analysis = {}
        if hasattr(env.envs[0], 'get_signal_weights_analysis'):
            weights_analysis = env.envs[0].get_signal_weights_analysis()
            logger.info(f"信号权重分析: {weights_analysis}")
        
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
        
        # 添加权重分析到训练配置
        training_config = {
            'algorithm': algorithm,
            'env_name': env_name,
            'total_timesteps': total_timesteps,
            'model_path': model_path,
            'portfolio_stats': portfolio_stats_serializable,
            'signal_weights_analysis': weights_analysis
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
        
        # 在评估结果中添加权重分析
        evaluation_results['signal_weights_analysis'] = weights_analysis
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(to_serializable(evaluation_results), f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {eval_file}")
        
        # 保存权重历史数据（如果可用）
        if hasattr(env.envs[0], 'signal_weights_history') and env.envs[0].signal_weights_history:
            weights_history = env.envs[0].signal_weights_history
            weights_file = f"{results_dir}/{algo}_{env_name}_weights_history.json"
            with open(weights_file, 'w', encoding='utf-8') as f:
                json.dump(to_serializable(weights_history), f, ensure_ascii=False, indent=2)
            logger.info(f"权重历史数据已保存到: {weights_file}")
        
        return {
            'model': model,
            'model_path': model_path,
            'evaluation_results': evaluation_results,
            'training_config': {
                'algorithm': algorithm,
                'env_type': env_type,
                'total_timesteps': total_timesteps
            },
            'weights_history': weights_history,
            'signal_names': signal_names,
            'final_weights_data': final_weights_data
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
            axes[0, 0].set_title('夏普比率比较')
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
    
    def plot_signal_weights_analysis(self, weights_history: List, signal_names: List[str] = None, save_path: str = None):
        """绘制信号权重分析图表
        
        Args:
            weights_history: 权重历史数据
            signal_names: 信号名称列表
            save_path: 保存路径
        """
        if not weights_history:
            logger.warning("没有权重历史数据可供分析")
            return
        
        if save_path is None:
            save_path = f"{self.storage_config['results_path']}/signal_weights_analysis.png"
        
        weights_array = np.array(weights_history)
        
        # 如果没有提供信号名称，使用默认名称
        if signal_names is None:
            signal_names = [f"Signal_{i+1}" for i in range(weights_array.shape[1])]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 权重时间序列图
        for i, signal_name in enumerate(signal_names):
            if i < weights_array.shape[1]:
                axes[0, 0].plot(weights_array[:, i], label=signal_name, alpha=0.8)
        axes[0, 0].set_title('信号权重时间序列')
        axes[0, 0].set_xlabel('训练步数')
        axes[0, 0].set_ylabel('权重值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 权重分布直方图
        for i, signal_name in enumerate(signal_names):
            if i < weights_array.shape[1]:
                axes[0, 1].hist(weights_array[:, i], bins=30, alpha=0.7, label=signal_name)
        axes[0, 1].set_title('信号权重分布')
        axes[0, 1].set_xlabel('权重值')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 平均权重柱状图
        avg_weights = np.mean(weights_array, axis=0)
        valid_signals = signal_names[:len(avg_weights)]
        axes[1, 0].bar(valid_signals, avg_weights, alpha=0.8)
        axes[1, 0].set_title('平均信号权重')
        axes[1, 0].set_xlabel('信号名称')
        axes[1, 0].set_ylabel('平均权重')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 权重相关性热力图
        if weights_array.shape[1] > 1:
            corr_matrix = np.corrcoef(weights_array.T)
            im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 1].set_title('权重相关性热力图')
            axes[1, 1].set_xticks(range(len(valid_signals)))
            axes[1, 1].set_yticks(range(len(valid_signals)))
            axes[1, 1].set_xticklabels(valid_signals, rotation=45)
            axes[1, 1].set_yticklabels(valid_signals)
            
            # 添加相关系数标签
            for i in range(len(valid_signals)):
                for j in range(len(valid_signals)):
                    text = axes[1, 1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                           ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"信号权重分析图表已保存到: {save_path}")
        
        # 打印权重统计信息
        logger.info("=== 信号权重统计信息 ===")
        for i, signal_name in enumerate(signal_names):
            if i < weights_array.shape[1]:
                weights = weights_array[:, i]
                logger.info(f"{signal_name}:")
                logger.info(f"  平均权重: {np.mean(weights):.4f}")
                logger.info(f"  标准差: {np.std(weights):.4f}")
                logger.info(f"  最小值: {np.min(weights):.4f}")
                logger.info(f"  最大值: {np.max(weights):.4f}")
                logger.info(f"  中位数: {np.median(weights):.4f}")

    def _record_final_model_weights(self, model: object, env: object, df: pd.DataFrame, 
                                   signal_names: List[str], final_model_dir: str, 
                                   algo: str, env_name: str) -> Dict:
        """记录最终模型在每个交易时段的权重分配
        
        Args:
            model: 训练好的模型
            env: 交易环境
            df: 数据DataFrame
            signal_names: 信号名称列表
            final_model_dir: 最终模型目录
            algo: 算法名称
            env_name: 环境名称
            
        Returns:
            权重记录数据
        """
        logger.info("开始记录最终模型在每个交易时段的权重分配...")
        
        # 创建权重记录列表
        weights_records = []
        
        # 重置环境
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # 遍历每个交易时段
        for step in range(len(df)):
            try:
                # 使用模型预测动作（权重）
                action, _ = model.predict(obs, deterministic=True)
                
                # 获取当前时间步的信息
                current_data = df.iloc[step]
                date_str = str(current_data.get('date', f'step_{step}'))
                
                # 记录权重信息
                weight_record = {
                    'step': step,
                    'date': date_str,
                    'weights': action.tolist() if hasattr(action, 'tolist') else action,
                    'signal_names': signal_names,
                    'portfolio_value': env.envs[0].get_portfolio_value(current_data['close']) if hasattr(env.envs[0], 'get_portfolio_value') else None,
                    'current_price': float(current_data['close']),
                    'position': env.envs[0].shares if hasattr(env.envs[0], 'shares') else 0,
                    'balance': env.envs[0].balance if hasattr(env.envs[0], 'balance') else 0
                }
                
                weights_records.append(weight_record)
                
                # 执行一步环境
                step_result = env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, done, truncated, info = step_result
                    done = done or truncated
                
                # 如果环境结束，跳出循环
                if done:
                    break
                    
            except Exception as e:
                logger.warning(f"记录第 {step} 步权重时出错: {e}")
                continue
        
        # 创建权重记录数据结构
        final_weights_data = {
            'model_info': {
                'algorithm': algo,
                'env_name': env_name,
                'training_completed_at': pd.Timestamp.now().isoformat(),
                'total_trading_steps': len(weights_records),
                'signal_names': signal_names
            },
            'weights_records': weights_records,
            'summary_stats': self._calculate_weights_summary_stats(weights_records, signal_names)
        }
        
        # 保存到JSON文件
        weights_file_path = f"{final_model_dir}/{algo}_{env_name}_final_weights.json"
        import json
        
        def to_serializable(val):
            if isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, (np.integer, np.floating)):
                return val.item()
            elif isinstance(val, pd.Timestamp):
                return val.isoformat()
            elif isinstance(val, dict):
                return {k: to_serializable(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [to_serializable(v) for v in val]
            else:
                return val
        
        with open(weights_file_path, 'w', encoding='utf-8') as f:
            json.dump(to_serializable(final_weights_data), f, ensure_ascii=False, indent=2)
        
        logger.info(f"最终模型权重记录已保存到: {weights_file_path}")
        logger.info(f"共记录了 {len(weights_records)} 个交易时段的权重分配")
        
        return final_weights_data
    
    def _calculate_weights_summary_stats(self, weights_records: List[Dict], signal_names: List[str]) -> Dict:
        """计算权重汇总统计信息
        
        Args:
            weights_records: 权重记录列表
            signal_names: 信号名称列表
            
        Returns:
            汇总统计信息
        """
        if not weights_records or not signal_names:
            return {}
        
        # 提取权重数据
        weights_data = []
        for record in weights_records:
            if 'weights' in record and isinstance(record['weights'], list):
                weights_data.append(record['weights'])
        
        if not weights_data:
            return {}
        
        weights_array = np.array(weights_data)
        
        # 计算统计信息
        summary_stats = {
            'signal_weights_summary': {},
            'overall_stats': {
                'total_trading_steps': len(weights_records),
                'avg_portfolio_value': np.mean([r.get('portfolio_value', 0) for r in weights_records if r.get('portfolio_value') is not None]),
                'final_portfolio_value': weights_records[-1].get('portfolio_value', 0) if weights_records else 0,
                'total_trades': len([r for r in weights_records if r.get('position', 0) != 0])
            }
        }
        
        # 为每个信号计算统计信息
        for i, signal_name in enumerate(signal_names):
            if i < weights_array.shape[1]:
                signal_weights = weights_array[:, i]
                summary_stats['signal_weights_summary'][signal_name] = {
                    'mean_weight': float(np.mean(signal_weights)),
                    'std_weight': float(np.std(signal_weights)),
                    'min_weight': float(np.min(signal_weights)),
                    'max_weight': float(np.max(signal_weights)),
                    'median_weight': float(np.median(signal_weights)),
                    'weight_stability': float(np.std(signal_weights) / (np.mean(signal_weights) + 1e-8))  # 变异系数
                }
        
        # 计算权重相关性
        if weights_array.shape[1] > 1:
            corr_matrix = np.corrcoef(weights_array.T)
            summary_stats['weights_correlation'] = {}
            for i in range(len(signal_names)):
                for j in range(i+1, len(signal_names)):
                    if i < corr_matrix.shape[0] and j < corr_matrix.shape[1]:
                        summary_stats['weights_correlation'][f"{signal_names[i]}_vs_{signal_names[j]}"] = float(corr_matrix[i, j])
        
        return summary_stats

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