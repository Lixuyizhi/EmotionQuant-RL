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
    def __init__(self, df: pd.DataFrame, config_path: str = "config/config.yaml", env_kwargs: dict = None, **kwargs):
        super().__init__()
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # 读取默认参数
        # 优先从model_training.max_weight_env，否则从backtest.env_params.max_weight_env
        file_config = config.get('model_training', {}).get('max_weight_env', {})
        if not file_config:
            file_config = config.get('backtest', {}).get('env_params', {}).get('max_weight_env', {})
        env_config = dict(file_config)
        if env_kwargs:
            env_config.update(env_kwargs)
        env_config.update(kwargs)

        self.df = df.copy()
        self.current_step = 0
        self.max_steps = len(df) - 1
        
        # 基础交易参数
        self.initial_balance = env_config.get('initial_balance', 1000000)
        self.transaction_fee = env_config.get('transaction_fee', 0.001)
        self.slippage = env_config.get('slippage', 0.0002)
        self.position_size = env_config.get('position_size', 0.05)
        self.max_position_ratio = env_config.get('max_position_ratio', 0.8)
        self.min_trade_amount = env_config.get('min_trade_amount', 5000)
        self.buy_threshold = env_config.get('buy_threshold', 0.1)
        self.sell_threshold = env_config.get('sell_threshold', -0.1)
        self.reward_scale = env_config.get('reward_scale', 1.0)
        self.risk_penalty = env_config.get('risk_penalty', 0.005)
        
        # 观察空间配置
        observation_config = env_config.get('observation_features', {})
        self.include_raw_ohlc = observation_config.get('include_raw_ohlc', False)
        self.include_normalized_ohlc = observation_config.get('include_normalized_ohlc', True)
        self.include_price_changes = observation_config.get('include_price_changes', True)
        self.include_volume = observation_config.get('include_volume', False)
        self.include_technical_indicators = observation_config.get('include_technical_indicators', True)
        self.include_account_info = observation_config.get('include_account_info', True)

        # 交易信号配置
        trading_config = env_config.get('trading_signals', {})
        self.enabled_signals = trading_config.get('enabled_signals', ['RSI_signal', 'BB_signal', 'SMA_signal', 'MACD_signal'])
        self.weight_min = trading_config.get('weight_min', 0.0)
        self.weight_max = trading_config.get('weight_max', 1.0)
        
        # 信号因子列
        self.signal_columns = self.enabled_signals
        # OHLC归一化列名
        self.ohlc_columns = ['norm_open', 'norm_high', 'norm_low', 'norm_close']
        # 价格变化率列名
        self.price_change_columns = ['pct_open', 'pct_high', 'pct_low', 'pct_close']
        # 原始OHLC列名
        self.raw_ohlc_columns = ['open', 'high', 'low', 'close']
        # 成交量列名
        self.volume_columns = ['volume', 'norm_volume'] if 'volume' in self.df.columns else []
        
        self._setup_spaces()
        self.trades = []
        self.portfolio_values = []
        self.weights_history = []

    def _setup_spaces(self):
        # 动作空间：动态权重数量，根据启用的信号数量
        self.action_space = spaces.Box(low=self.weight_min, high=self.weight_max, shape=(len(self.enabled_signals),), dtype=np.float32)
        
        # 计算观察空间大小
        obs_size = 0
        
        # 交易信号（动态数量）
        if self.include_technical_indicators:
            obs_size += len(self.enabled_signals)  # 动态信号数量
        
        # 账户信息
        if self.include_account_info:
            obs_size += 3  # shares_held, balance, portfolio_value
        
        # OHLC归一化数据
        if self.include_normalized_ohlc:
            obs_size += len(self.ohlc_columns)
        
        # 价格变化率
        if self.include_price_changes:
            obs_size += len(self.price_change_columns)
        
        # 原始OHLC数据
        if self.include_raw_ohlc:
            obs_size += len(self.raw_ohlc_columns)
        
        # 成交量数据
        if self.include_volume:
            obs_size += len(self.volume_columns)
        
        # 观察空间：动态大小的特征向量
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
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
            # 如果所有权重都为0，使用等权重
            return np.array([1.0/len(self.enabled_signals)] * len(self.enabled_signals), dtype=np.float32)

    def _get_current_signals(self) -> np.ndarray:
        """获取当前交易信号（使用配置中启用的信号）"""
        if self.current_step >= len(self.df):
            return np.zeros(len(self.enabled_signals), dtype=np.float32)
        signals = []
        
        for signal_col in self.enabled_signals:
            if signal_col in self.df.columns:
                signal_value = self.df.iloc[self.current_step][signal_col]
                if pd.isna(signal_value):
                    signal_value = 0
                signals.append(signal_value)
            else:
                signals.append(0)
        
        # 确保返回正确数量的信号
        while len(signals) < len(self.enabled_signals):
            signals.append(0)
        
        return np.array(signals[:len(self.enabled_signals)], dtype=np.float32)

    def _get_observation_signals(self) -> np.ndarray:
        """获取观察空间中的交易信号（与交易决策保持一致）"""
        return self._get_current_signals()

    def _signal_to_action(self, signal: float) -> int:
        """将信号转换为交易动作
        
        Args:
            signal: 交易信号值
            
        Returns:
            -1: 卖出, 0: 持有, 1: 买入
        """
        if signal > self.buy_threshold:  # 使用配置的买入阈值
            return 1  # 买入
        elif signal < self.sell_threshold:  # 使用配置的卖出阈值
            return -1  # 卖出
        else:
            return 0  # 持有

    def _execute_trade(self, action: int, current_price: float) -> float:
        portfolio_value_before = self._get_portfolio_value(current_price)
        
        # 记录交易前的状态
        shares_before = self.shares_held
        balance_before = self.balance
        
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
        
        # 计算基础奖励（组合价值变化）
        portfolio_value_after = self._get_portfolio_value(current_price)
        base_reward = portfolio_value_after - portfolio_value_before
        
        # 改进的奖励函数设计
        reward = self._calculate_improved_reward(
            base_reward, action, current_price, 
            shares_before, balance_before, 
            portfolio_value_before, portfolio_value_after
        )
        
        return reward

    def _calculate_improved_reward(self, base_reward: float, action: int, current_price: float,
                                 shares_before: int, balance_before: float,
                                 portfolio_value_before: float, portfolio_value_after: float) -> float:
        """计算改进的奖励函数 - 保守策略版本
        
        包含以下组件：
        1. 基础收益奖励
        2. 交易激励奖励
        3. 风险调整奖励
        4. 持仓平衡奖励
        5. 风险惩罚
        6. 交易质量奖励
        """
        reward = 0.0
        
        # 1. 基础收益奖励（应用缩放）
        reward += base_reward * self.reward_scale
        
        # 2. 交易激励奖励（大幅降低，鼓励谨慎交易）
        if action != 0:  # 如果有交易行为
            # 交易激励：给予很小的正向奖励
            trade_incentive = 10  # 大幅降低基础交易激励
            reward += trade_incentive
            
            # 如果交易带来收益，给予额外奖励
            if base_reward > 0:
                profit_bonus = base_reward * 0.5  # 增加收益奖励比例
                reward += profit_bonus
            else:
                # 亏损交易给予较大惩罚
                loss_penalty = base_reward * 0.3  # 增加亏损惩罚
                reward += loss_penalty
        
        # 3. 风险调整奖励（降低放大倍数）
        if len(self.portfolio_values) > 1:
            # 计算收益率
            returns = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
            # 简单的风险调整：收益率越高，奖励越大
            if returns > 0:
                risk_adjusted_bonus = returns * 200  # 大幅降低放大倍数
                reward += risk_adjusted_bonus
        
        # 4. 持仓平衡奖励（鼓励适度持仓）
        current_position_ratio = (self.shares_held * current_price) / self.initial_balance
        optimal_position_ratio = 0.3  # 降低理想持仓比例，更加保守
        
        # 持仓平衡奖励：越接近理想持仓，奖励越高
        position_balance = 1.0 - abs(current_position_ratio - optimal_position_ratio)
        position_bonus = position_balance * 15  # 降低持仓平衡奖励
        reward += position_bonus
        
        # 5. 风险惩罚（增加惩罚强度）
        if current_position_ratio > self.max_position_ratio:
            risk_penalty = (current_position_ratio - self.max_position_ratio) * self.risk_penalty
            reward -= risk_penalty
        
        # 6. 空仓惩罚（减少空仓惩罚，允许空仓）
        if current_position_ratio < 0.02 and action == 0:  # 降低空仓阈值
            empty_position_penalty = -5  # 减少空仓惩罚
            reward += empty_position_penalty
        
        # 7. 过度交易惩罚（大幅增加惩罚）
        if len(self.trades) > 0:
            recent_trades = [t for t in self.trades if t['step'] >= self.current_step - 30]
            if len(recent_trades) > 5:  # 大幅降低过度交易限制
                overtrading_penalty = -50  # 大幅增加过度交易惩罚
                reward += overtrading_penalty
        
        # 8. 新增：交易质量奖励（增加质量奖励）
        if action != 0 and len(self.trades) > 0:
            # 检查最近交易的盈利情况
            recent_profitable_trades = 0
            recent_trades = [t for t in self.trades if t['step'] >= self.current_step - 15]
            
            for trade in recent_trades:
                if trade['action'] == 'buy':
                    # 买入后价格上涨
                    if current_price > trade['price']:
                        recent_profitable_trades += 1
                elif trade['action'] == 'sell':
                    # 卖出后价格下跌
                    if current_price < trade['price']:
                        recent_profitable_trades += 1
            
            # 交易质量奖励
            if len(recent_trades) > 0:
                success_rate = recent_profitable_trades / len(recent_trades)
                quality_bonus = success_rate * 30  # 增加成功交易奖励
                reward += quality_bonus
        
        # 9. 新增：连续亏损惩罚
        if len(self.portfolio_values) > 1:
            # 检查最近几步是否连续亏损
            recent_values = self.portfolio_values[-5:] if len(self.portfolio_values) >= 5 else self.portfolio_values
            if len(recent_values) >= 3:
                consecutive_losses = 0
                for i in range(1, len(recent_values)):
                    if recent_values[i] < recent_values[i-1]:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                
                if consecutive_losses >= 3:  # 连续3次亏损
                    consecutive_loss_penalty = -20
                    reward += consecutive_loss_penalty
        
        return reward

    def _get_portfolio_value(self, current_price: float) -> float:
        return self.balance + (self.shares_held * current_price)

    def _get_observation(self) -> np.ndarray:
        """获取观察（动态特征组合）"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape[0])
        
        observation_parts = []
        
        # 获取当前交易信号
        if self.include_technical_indicators:
            current_signals = self._get_observation_signals()
            observation_parts.append(current_signals)
        
        # 获取账户信息
        if self.include_account_info:
            current_price = self.df.iloc[self.current_step]['close']
            account_info = np.array([
                self.shares_held,  # 当前持仓
                self.balance,      # 当前余额
                self._get_portfolio_value(current_price)  # 当前组合价值
            ])
            observation_parts.append(account_info)
        
        # 获取OHLC归一化数据
        if self.include_normalized_ohlc:
            current_ohlc = []
            for ohlc_col in self.ohlc_columns:
                if ohlc_col in self.df.columns:
                    ohlc_value = self.df.iloc[self.current_step][ohlc_col]
                    if pd.isna(ohlc_value):
                        ohlc_value = 1.0  # 默认值
                    current_ohlc.append(ohlc_value)
                else:
                    current_ohlc.append(1.0)  # 默认值
            observation_parts.append(np.array(current_ohlc))
        
        # 获取价格变化率数据
        if self.include_price_changes:
            current_price_changes = []
            for pct_col in self.price_change_columns:
                if pct_col in self.df.columns:
                    pct_value = self.df.iloc[self.current_step][pct_col]
                    if pd.isna(pct_value):
                        pct_value = 0.0  # 默认值
                    current_price_changes.append(pct_value)
                else:
                    current_price_changes.append(0.0)  # 默认值
            observation_parts.append(np.array(current_price_changes))
        
        # 获取原始OHLC数据
        if self.include_raw_ohlc:
            current_raw_ohlc = []
            for ohlc_col in self.raw_ohlc_columns:
                if ohlc_col in self.df.columns:
                    ohlc_value = self.df.iloc[self.current_step][ohlc_col]
                    if pd.isna(ohlc_value):
                        ohlc_value = 0.0  # 默认值
                    current_raw_ohlc.append(ohlc_value)
                else:
                    current_raw_ohlc.append(0.0)  # 默认值
            observation_parts.append(np.array(current_raw_ohlc))
        
        # 获取成交量数据
        if self.include_volume:
            current_volume = []
            for vol_col in self.volume_columns:
                if vol_col in self.df.columns:
                    vol_value = self.df.iloc[self.current_step][vol_col]
                    if pd.isna(vol_value):
                        vol_value = 0.0  # 默认值
                    current_volume.append(vol_value)
                else:
                    current_volume.append(0.0)  # 默认值
            observation_parts.append(np.array(current_volume))
        
        # 合并所有特征
        if observation_parts:
            observation = np.concatenate(observation_parts)
        else:
            observation = np.array([])
        
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