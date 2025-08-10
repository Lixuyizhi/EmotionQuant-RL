import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import logging
import yaml
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class SignalWeightTradingEnv(gym.Env):
    """交易信号权重环境
    强化学习模型学习给RSI_signal、BB_signal、SMA_signal分配权重
    然后根据加权总和决定交易方向
    """
    
    def __init__(self, df: pd.DataFrame, config_path: str = "config/config.yaml", env_kwargs: dict = None, **kwargs):
        super().__init__()
        # 使用配置管理器加载配置
        try:
            config_manager = ConfigManager(config_path)
            env_config = config_manager.get_env_config('signal_weight_env')
        except Exception as e:
            logger.warning(f"配置管理器加载失败，使用传统方式: {e}")
            # 回退到传统配置加载方式
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            env_config = config.get('model_training', {}).get('signal_weight_env', {})
        
        # 合并外部参数，优先用env_kwargs和kwargs
        if env_kwargs:
            env_config.update(env_kwargs)
        env_config.update(kwargs)

        self.df = df.copy()
        self.current_step = 0
        self.max_steps = len(df) - 1

        # 基础交易参数
        self.initial_balance = env_config.get('initial_balance', 100000)
        self.transaction_fee = env_config.get('transaction_fee', 0.001)
        self.slippage = env_config.get('slippage', 0.0005)
        self.position_size = env_config.get('position_size', 0.1)

        # 交易阈值参数
        self.buy_threshold = env_config.get('buy_threshold', 0.1)
        self.sell_threshold = env_config.get('sell_threshold', -0.1)

        # 风险控制参数
        self.max_position_ratio = env_config.get('max_position_ratio', 0.8)
        self.min_trade_amount = env_config.get('min_trade_amount', 1000)

        # 奖励函数参数
        self.reward_scale = env_config.get('reward_scale', 1.0)
        self.risk_penalty = env_config.get('risk_penalty', 0.01)

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
        
        # 交易信号列名
        self.signal_columns = self.enabled_signals
        
        # OHLC归一化列名
        self.ohlc_columns = ['norm_open', 'norm_high', 'norm_low', 'norm_close']
        
        # 价格变化率列名
        self.price_change_columns = ['pct_open', 'pct_high', 'pct_low', 'pct_close']
        
        # 原始OHLC列名
        self.raw_ohlc_columns = ['open', 'high', 'low', 'close']
        
        # 成交量列名
        self.volume_columns = ['volume', 'norm_volume'] if 'volume' in self.df.columns else []

        # 其他特征列（不包括交易信号）
        self.feature_columns = [col for col in self.df.columns 
                               if col not in ['date', 'open', 'high', 'low', 'close', 'volume'] 
                               and not col.startswith('target_')
                               and col not in self.signal_columns
                               and col not in self.ohlc_columns
                               and col not in self.price_change_columns
                               and col not in self.volume_columns]

        self._setup_spaces()
        self.trades = []
        self.portfolio_values = []
        self.weights_history = []

    def _setup_spaces(self):
        """设置观察空间和动作空间"""
        # 动作空间：动态权重值（根据启用的信号数量）
        # 权重范围：[weight_min, weight_max]，总和为1
        self.action_space = spaces.Box(
            low=self.weight_min,
            high=self.weight_max,
            shape=(len(self.enabled_signals),),  # 动态信号数量
            dtype=np.float32
        )
        
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
        self.weights_history = []
        
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        """执行一步交易
        
        Args:
            action: 动态数量的权重值
        """
        # 归一化权重，确保总和为1
        weights = self._normalize_weights(action)
        self.weights_history.append(weights.copy())
        
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
            return np.array([1.0/len(self.enabled_signals)] * len(self.enabled_signals), dtype=np.float32)

    def _get_current_signals(self) -> np.ndarray:
        """获取当前交易信号（使用配置中启用的信号）"""
        if self.current_step >= len(self.df):
            return np.zeros(len(self.enabled_signals), dtype=np.float32)
        
        signals = []
        
        for signal_col in self.enabled_signals:
            if signal_col in self.df.columns:
                signal_value = self.df.iloc[self.current_step][signal_col]
                # 处理NaN值
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
        
        # 记录交易前的状态
        shares_before = self.shares_held
        balance_before = self.balance
        
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
                        'value': sell_value
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
        """计算组合价值"""
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
            'avg_signal_weights': np.mean(self.weights_history, axis=0) if self.weights_history else [0.0] * len(self.enabled_signals)
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
        if not self.weights_history:
            return {}
        
        weights_array = np.array(self.weights_history)
        
        analysis = {
            'avg_weights': {},
            'std_weights': {},
            'max_weights': {},
            'min_weights': {}
        }
        
        # 动态生成权重分析，基于启用的信号
        for i, signal_name in enumerate(self.enabled_signals):
            if i < weights_array.shape[1]:  # 确保索引有效
                analysis['avg_weights'][signal_name] = float(np.mean(weights_array[:, i]))
                analysis['std_weights'][signal_name] = float(np.std(weights_array[:, i]))
                analysis['max_weights'][signal_name] = float(np.max(weights_array[:, i]))
                analysis['min_weights'][signal_name] = float(np.min(weights_array[:, i]))
        
        return analysis 