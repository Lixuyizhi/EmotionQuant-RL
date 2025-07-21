import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import logging
import yaml

logger = logging.getLogger(__name__)

class MaxWeightTradingEnv(gym.Env):
    """最大权重信号交易环境：模型输出权重，实际只用权重最大的因子信号做交易决策"""
    def __init__(self, df: pd.DataFrame, config_path: str = "config/config.yaml"):
        super().__init__()
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.df = df.copy()
        self.current_step = 0
        self.max_steps = len(df) - 1
        # 信号因子列
        self.signal_columns = ['RSI_signal', 'BB_signal', 'SMA_signal']
        self.initial_balance = self.config['strategy']['max_weight_strategy']['initial_balance']
        self.transaction_fee = self.config['strategy']['max_weight_strategy']['transaction_fee']
        self.slippage = self.config['strategy']['max_weight_strategy']['slippage']
        self.position_size = self.config['strategy']['max_weight_strategy']['position_size']
        self._setup_spaces()
        self.trades = []
        self.portfolio_values = []
        self.weights_history = []

    def _setup_spaces(self):
        # 动作空间：3个连续权重
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        # 观察空间：3个信号 + 账户信息
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),  # 3信号+3账户
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
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
        self.weights_history = []
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        # 归一化权重
        weights = self._normalize_weights(action)
        self.weights_history.append(weights.copy())
        # 获取当前信号
        current_signals = self._get_current_signals()
        # 只用权重最大的信号做决策
        max_idx = np.argmax(weights)
        chosen_signal = current_signals[max_idx]
        # 信号转动作
        trade_action = self._signal_to_action(chosen_signal)
        current_price = self.df.iloc[self.current_step]['close']
        reward = self._execute_trade(trade_action, current_price)
        self.current_step += 1
        portfolio_value = self._get_portfolio_value(current_price)
        self.portfolio_values.append(portfolio_value)
        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = self._get_observation()
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'step': self.current_step,
            'weights': weights,
            'chosen_signal': chosen_signal
        }
        return obs, reward, terminated, truncated, info

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        total = np.sum(weights)
        if total > 0:
            return weights / total
        else:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    def _get_current_signals(self) -> np.ndarray:
        if self.current_step >= len(self.df):
            return np.zeros(3, dtype=np.float32)
        signals = []
        for signal_col in self.signal_columns:
            if signal_col in self.df.columns:
                signal_value = self.df.iloc[self.current_step][signal_col]
                if pd.isna(signal_value):
                    signal_value = 0
                signals.append(signal_value)
            else:
                signals.append(0)
        return np.array(signals, dtype=np.float32)

    def _signal_to_action(self, signal: float) -> int:
        if signal > 0.1:
            return 1  # 买入
        elif signal < -0.1:
            return -1  # 卖出
        else:
            return 0  # 持有

    def _execute_trade(self, action: int, current_price: float) -> float:
        portfolio_value_before = self._get_portfolio_value(current_price)
        if action == -1:  # 卖出
            if self.shares_held > 0:
                shares_to_sell = min(self.shares_held, int(self.shares_held * self.position_size))
                if shares_to_sell > 0:
                    sell_price = current_price * (1 - self.slippage - self.transaction_fee)
                    sell_value = shares_to_sell * sell_price
                    self.balance += sell_value
                    self.shares_held -= shares_to_sell
                    self.total_shares_sold += shares_to_sell
                    self.total_sales_value += sell_value
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'sell',
                        'shares': shares_to_sell,
                        'price': sell_price,
                        'value': sell_value
                    })
        elif action == 1:  # 买入
            if self.balance > 0:
                max_shares = int(self.balance / (current_price * (1 + self.slippage + self.transaction_fee)))
                shares_to_buy = min(max_shares, int(max_shares * self.position_size))
                if shares_to_buy > 0:
                    buy_price = current_price * (1 + self.slippage + self.transaction_fee)
                    buy_value = shares_to_buy * buy_price
                    if buy_value <= self.balance:
                        self.balance -= buy_value
                        self.shares_held += shares_to_buy
                        self.total_shares_bought += shares_to_buy
                        self.total_buy_value += buy_value
                        self.trades.append({
                            'step': self.current_step,
                            'action': 'buy',
                            'shares': shares_to_buy,
                            'price': buy_price,
                            'value': buy_value
                        })
        portfolio_value_after = self._get_portfolio_value(current_price)
        reward = portfolio_value_after - portfolio_value_before
        return reward

    def _get_portfolio_value(self, current_price: float) -> float:
        return self.balance + (self.shares_held * current_price)

    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape[0])
        current_signals = self._get_current_signals()
        observation = np.concatenate([
            current_signals,
            [self.shares_held],
            [self.balance],
            [self._get_portfolio_value(self.df.iloc[self.current_step]['close'])]
        ])
        return observation.astype(np.float32)

    def get_portfolio_stats(self) -> Dict:
        if not self.portfolio_values:
            return {}
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        stats = {
            'total_return': (portfolio_values[-1] - self.initial_balance) / self.initial_balance,
            'final_value': portfolio_values[-1],
            'max_value': np.max(portfolio_values),
            'min_value': np.min(portfolio_values),
            'volatility': np.std(returns) * np.sqrt(252),
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'total_trades': len(self.trades)
        }
        return stats

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd 