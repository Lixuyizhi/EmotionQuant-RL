import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import logging
import yaml

logger = logging.getLogger(__name__)

class OilTradingEnv(gym.Env):
    """原油期货交易环境，兼容gymnasium"""
    
    def __init__(self, df: pd.DataFrame, config_path: str = "config/config.yaml"):
        super().__init__()
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.df = df.copy()
        self.current_step = 0
        self.max_steps = len(df) - 1
        self.initial_balance = self.config['strategy']['weighted_strategy']['initial_balance']
        self.transaction_fee = self.config['strategy']['weighted_strategy']['transaction_fee']
        self.slippage = self.config['strategy']['weighted_strategy']['slippage']
        self.position_size = self.config['strategy']['weighted_strategy']['position_size']
        
        self._setup_spaces()
        self.trades = []
        self.portfolio_values = []

    def _setup_spaces(self):
        feature_columns = [col for col in self.df.columns 
                          if col not in ['date', 'open', 'high', 'low', 'close', 'volume'] 
                          and not col.startswith('target_')]
        self.n_features = len(feature_columns)
        self.feature_columns = feature_columns
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_features + 4,),
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
        
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: int):
        current_price = self.df.iloc[self.current_step]['close']
        
        reward = self._execute_trade(action, current_price)
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
            'step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info

    def _execute_trade(self, action: int, current_price: float) -> float:
        portfolio_value_before = self._get_portfolio_value(current_price)
        if action == 0:  # 卖出
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
        elif action == 2:  # 买入
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
        features = self.df.iloc[self.current_step][self.feature_columns].values
        current_price = self.df.iloc[self.current_step]['close']
        observation = np.concatenate([
            features,
            [self.shares_held],
            [self.balance],
            [current_price],
            [self._get_portfolio_value(current_price)]
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

class MultiFactorTradingEnv(OilTradingEnv):
    def __init__(self, df: pd.DataFrame, feature_weights: Dict[str, float], config_path: str = "config/config.yaml"):
        super().__init__(df, config_path)
        self.feature_weights = feature_weights
        
    def _get_weighted_action(self, observation: np.ndarray) -> int:
        features = observation[:self.n_features]
        action_signals = {}
        weighted_sum = 0
        for feature_name, weight in self.feature_weights.items():
            if feature_name in self.feature_columns:
                feature_idx = self.feature_columns.index(feature_name)
                feature_value = features[feature_idx]
                if 'RSI' in feature_name:
                    if feature_value > 70:
                        signal = -1
                    elif feature_value < 30:
                        signal = 1
                    else:
                        signal = 0
                elif 'MACD' in feature_name:
                    signal = 1 if feature_value > 0 else -1
                elif 'BB' in feature_name:
                    signal = -1 if feature_value > 0.8 else (1 if feature_value < 0.2 else 0)
                elif 'momentum' in feature_name:
                    signal = 1 if feature_value > 0.02 else (-1 if feature_value < -0.02 else 0)
                else:
                    signal = 1 if feature_value > 0 else (-1 if feature_value < 0 else 0)
                action_signals[feature_name] = signal
                weighted_sum += signal * weight
        if weighted_sum > 0.1:
            return 2
        elif weighted_sum < -0.1:
            return 0
        else:
            return 1

class MaxWeightTradingEnv(OilTradingEnv):
    def __init__(self, df: pd.DataFrame, feature_weights: Dict[str, float], config_path: str = "config/config.yaml"):
        super().__init__(df, config_path)
        self.feature_weights = feature_weights
        
    def _get_max_weight_action(self, observation: np.ndarray) -> int:
        features = observation[:self.n_features]
        max_weight = 0
        best_action = 1  # 默认持有
        
        for feature_name, weight in self.feature_weights.items():
            if feature_name in self.feature_columns:
                feature_idx = self.feature_columns.index(feature_name)
                feature_value = features[feature_idx]
                
                # 根据特征值确定动作
                if 'RSI' in feature_name:
                    if feature_value > 70:
                        action = 0  # 卖出
                    elif feature_value < 30:
                        action = 2  # 买入
                    else:
                        action = 1  # 持有
                elif 'MACD' in feature_name:
                    action = 2 if feature_value > 0 else 0
                elif 'BB' in feature_name:
                    if feature_value > 0.8:
                        action = 0
                    elif feature_value < 0.2:
                        action = 2
                    else:
                        action = 1
                else:
                    action = 2 if feature_value > 0 else (0 if feature_value < 0 else 1)
                
                # 选择权重最大的动作
                if weight > max_weight:
                    max_weight = weight
                    best_action = action
        
        return best_action 