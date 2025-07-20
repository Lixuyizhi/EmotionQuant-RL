import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import logging
import yaml

logger = logging.getLogger(__name__)

class SignalWeightTradingEnv(gym.Env):
    """交易信号权重环境
    强化学习模型学习给RSI_signal、BB_signal、SMA_signal分配权重
    然后根据加权总和决定交易方向
    """
    
    def __init__(self, df: pd.DataFrame, config_path: str = "config/config.yaml"):
        super().__init__()
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.df = df.copy()
        self.current_step = 0
        self.max_steps = len(df) - 1
        
        # 从配置文件读取信号权重环境参数
        env_config = self.config.get('model_training', {}).get('signal_weight_env', {})
        
        # 基础交易参数
        self.initial_balance = env_config.get('initial_balance', 100000)
        self.transaction_fee = env_config.get('transaction_fee', 0.001)  # 0.1%
        self.slippage = env_config.get('slippage', 0.0005)  # 0.05%
        self.position_size = env_config.get('position_size', 0.1)  # 10%
        
        # 交易阈值参数
        self.buy_threshold = env_config.get('buy_threshold', 0.1)  # 买入阈值
        self.sell_threshold = env_config.get('sell_threshold', -0.1)  # 卖出阈值
        
        # 风险控制参数
        self.max_position_ratio = env_config.get('max_position_ratio', 0.8)  # 最大仓位比例
        self.min_trade_amount = env_config.get('min_trade_amount', 1000)  # 最小交易金额
        
        # 奖励函数参数
        self.reward_scale = env_config.get('reward_scale', 1.0)  # 奖励缩放因子
        self.risk_penalty = env_config.get('risk_penalty', 0.01)  # 风险惩罚系数
        
        # 交易信号列名
        self.signal_columns = ['RSI_signal', 'BB_signal', 'SMA_signal']
        
        # 其他特征列（不包括交易信号）
        self.feature_columns = [col for col in self.df.columns 
                               if col not in ['date', 'open', 'high', 'low', 'close', 'volume'] 
                               and not col.startswith('target_')
                               and col not in self.signal_columns]
        
        self._setup_spaces()
        self.trades = []
        self.portfolio_values = []
        self.signal_weights_history = []

    def _setup_spaces(self):
        """设置观察空间和动作空间"""
        # 动作空间：3个权重值（RSI_signal, BB_signal, SMA_signal的权重）
        # 权重范围：[0, 1]，总和为1
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),  # 3个信号权重
            dtype=np.float32
        )
        
        # 观察空间：交易信号 + 账户信息（方案2）
        # RSI_signal, BB_signal, SMA_signal, shares_held, balance, portfolio_value
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),  # 3个信号 + 3个账户信息
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_shares_bought = 0
        self.total_buy_value = 0
        self.trades = []
        self.portfolio_values = []
        self.signal_weights_history = []
        
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        """执行一步交易
        
        Args:
            action: 3个权重值 [rsi_weight, bb_weight, sma_weight]
        """
        # 归一化权重，确保总和为1
        weights = self._normalize_weights(action)
        self.signal_weights_history.append(weights.copy())
        
        # 获取当前交易信号
        current_signals = self._get_current_signals()
        
        # 计算加权信号总和
        weighted_signal = np.sum(weights * current_signals)
        
        # 根据加权信号决定交易动作
        trade_action = self._signal_to_action(weighted_signal)
        
        # 执行交易
        current_price = self.df.iloc[self.current_step]['close']
        reward = self._execute_trade(trade_action, current_price)
        
        # 更新状态
        self.current_step += 1
        portfolio_value = self._get_portfolio_value(current_price)
        self.portfolio_values.append(portfolio_value)
        
        # 检查是否结束
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 获取观察
        obs = self._get_observation()
        
        # 额外信息
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'step': self.current_step,
            'weighted_signal': weighted_signal,
            'trade_action': trade_action,
            'signal_weights': weights,
            'current_signals': current_signals
        }
        
        return obs, reward, terminated, truncated, info

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """归一化权重，确保总和为1"""
        total = np.sum(weights)
        if total > 0:
            return weights / total
        else:
            # 如果所有权重都为0，使用等权重
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    def _get_current_signals(self) -> np.ndarray:
        """获取当前交易信号"""
        if self.current_step >= len(self.df):
            return np.array([0, 0, 0], dtype=np.float32)
        
        signals = []
        for signal_col in self.signal_columns:
            if signal_col in self.df.columns:
                signal_value = self.df.iloc[self.current_step][signal_col]
                # 处理NaN值
                if pd.isna(signal_value):
                    signal_value = 0
                signals.append(signal_value)
            else:
                signals.append(0)
        
        return np.array(signals, dtype=np.float32)

    def _signal_to_action(self, weighted_signal: float) -> int:
        """将加权信号转换为交易动作
        
        Args:
            weighted_signal: 加权信号总和
            
        Returns:
            -1: 卖出, 0: 持有, 1: 买入
        """
        if weighted_signal > self.buy_threshold:  # 买入阈值
            return 1  # 买入
        elif weighted_signal < self.sell_threshold:  # 卖出阈值
            return -1  # 卖出
        else:
            return 0  # 持有

    def _execute_trade(self, action: int, current_price: float) -> float:
        """执行交易（包含完整的手续费、滑点、风险控制）"""
        portfolio_value_before = self._get_portfolio_value(current_price)
        
        if action == -1:  # 卖出
            if self.shares_held > 0:
                # 计算卖出数量（考虑仓位大小）
                shares_to_sell = min(self.shares_held, int(self.shares_held * self.position_size))
                
                # 检查最小交易金额
                sell_value_estimate = shares_to_sell * current_price
                if sell_value_estimate >= self.min_trade_amount and shares_to_sell > 0:
                    # 计算实际卖出价格（考虑滑点和手续费）
                    sell_price = current_price * (1 - self.slippage - self.transaction_fee)
                    sell_value = shares_to_sell * sell_price
                    
                    # 执行卖出
                    self.balance += sell_value
                    self.shares_held -= shares_to_sell
                    self.total_shares_sold += shares_to_sell
                    self.total_sales_value += sell_value
                    
                    # 记录交易
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'sell',
                        'shares': shares_to_sell,
                        'price': sell_price,
                        'value': sell_value,
                        'transaction_fee': shares_to_sell * current_price * self.transaction_fee,
                        'slippage_cost': shares_to_sell * current_price * self.slippage
                    })
                    
        elif action == 1:  # 买入
            if self.balance > 0:
                # 计算最大可买数量（考虑手续费和滑点）
                max_shares = int(self.balance / (current_price * (1 + self.slippage + self.transaction_fee)))
                
                # 计算实际买入数量（考虑仓位大小和最大仓位限制）
                current_position_value = self.shares_held * current_price
                max_position_value = self.initial_balance * self.max_position_ratio
                
                if current_position_value < max_position_value:
                    shares_to_buy = min(max_shares, int(max_shares * self.position_size))
                    
                    # 检查最小交易金额
                    buy_value_estimate = shares_to_buy * current_price
                    if buy_value_estimate >= self.min_trade_amount and shares_to_buy > 0:
                        # 计算实际买入价格（考虑滑点和手续费）
                        buy_price = current_price * (1 + self.slippage + self.transaction_fee)
                        buy_value = shares_to_buy * buy_price
                        
                        # 检查余额是否足够
                        if buy_value <= self.balance:
                            # 执行买入
                            self.balance -= buy_value
                            self.shares_held += shares_to_buy
                            self.total_shares_bought += shares_to_buy
                            self.total_buy_value += buy_value
                            
                            # 记录交易
                            self.trades.append({
                                'step': self.current_step,
                                'action': 'buy',
                                'shares': shares_to_buy,
                                'price': buy_price,
                                'value': buy_value,
                                'transaction_fee': shares_to_buy * current_price * self.transaction_fee,
                                'slippage_cost': shares_to_buy * current_price * self.slippage
                            })
        
        portfolio_value_after = self._get_portfolio_value(current_price)
        reward = portfolio_value_after - portfolio_value_before
        
        # 应用奖励缩放和风险惩罚
        reward = reward * self.reward_scale
        
        # 添加风险惩罚（基于持仓比例）
        current_position_ratio = (self.shares_held * current_price) / self.initial_balance
        if current_position_ratio > self.max_position_ratio:
            risk_penalty = (current_position_ratio - self.max_position_ratio) * self.risk_penalty
            reward -= risk_penalty
        
        return reward

    def _get_portfolio_value(self, current_price: float) -> float:
        """计算组合价值"""
        return self.balance + (self.shares_held * current_price)

    def _get_observation(self) -> np.ndarray:
        """获取观察（方案2：只包含交易信号和账户信息）"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape[0])
        
        # 获取当前交易信号
        current_signals = self._get_current_signals()
        current_price = self.df.iloc[self.current_step]['close']
        
        # 构建观察向量：信号 + 账户信息
        observation = np.concatenate([
            current_signals,  # RSI_signal, BB_signal, SMA_signal
            [self.shares_held],  # 当前持仓
            [self.balance],      # 当前余额
            [self._get_portfolio_value(current_price)]  # 当前组合价值
        ])
        
        return observation.astype(np.float32)

    def get_portfolio_stats(self) -> Dict:
        """获取组合统计信息"""
        if not self.portfolio_values:
            return {}
        
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        stats = {
            'total_return': (portfolio_values[-1] - self.initial_balance) / self.initial_balance,
            'final_value': portfolio_values[-1],
            'max_value': np.max(portfolio_values),
            'min_value': np.min(portfolio_values),
            'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'total_trades': len(self.trades),
            'avg_signal_weights': np.mean(self.signal_weights_history, axis=0) if self.signal_weights_history else [0, 0, 0]
        }
        
        return stats

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """计算最大回撤"""
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def get_signal_weights_analysis(self) -> Dict:
        """获取信号权重分析"""
        if not self.signal_weights_history:
            return {}
        
        weights_array = np.array(self.signal_weights_history)
        
        analysis = {
            'avg_weights': {
                'RSI_signal': float(np.mean(weights_array[:, 0])),
                'BB_signal': float(np.mean(weights_array[:, 1])),
                'SMA_signal': float(np.mean(weights_array[:, 2]))
            },
            'std_weights': {
                'RSI_signal': float(np.std(weights_array[:, 0])),
                'BB_signal': float(np.std(weights_array[:, 1])),
                'SMA_signal': float(np.std(weights_array[:, 2]))
            },
            'max_weights': {
                'RSI_signal': float(np.max(weights_array[:, 0])),
                'BB_signal': float(np.max(weights_array[:, 1])),
                'SMA_signal': float(np.max(weights_array[:, 2]))
            },
            'min_weights': {
                'RSI_signal': float(np.min(weights_array[:, 0])),
                'BB_signal': float(np.min(weights_array[:, 1])),
                'SMA_signal': float(np.min(weights_array[:, 2]))
            }
        }
        
        return analysis 