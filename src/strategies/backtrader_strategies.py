import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseStrategy(bt.Strategy):
    """基础交易策略类"""
    
    def __init__(self):
        """初始化策略"""
        super().__init__()
        
        # 技术指标
        self.sma_20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        self.sma_50 = bt.indicators.SimpleMovingAverage(self.data.close, period=50)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.macd = bt.indicators.MACD(self.data.close)
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20)
        
        # 交易状态
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # 交易统计
        self.trades = []
        self.portfolio_values = []
        
    def log(self, txt, dt=None):
        """记录日志"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()}, {txt}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行, 价格: {order.executed.price:.2f}, '
                        f'成本: {order.executed.value:.2f}, '
                        f'手续费: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'卖出执行, 价格: {order.executed.price:.2f}, '
                        f'成本: {order.executed.value:.2f}, '
                        f'手续费: {order.executed.comm:.2f}')
            
            # 记录交易
            self.trades.append({
                'date': self.datas[0].datetime.date(0),
                'action': 'buy' if order.isbuy() else 'sell',
                'price': order.executed.price,
                'size': order.executed.size,
                'value': order.executed.value,
                'commission': order.executed.comm
            })
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易完成通知"""
        if not trade.isclosed:
            return
        
        self.log(f'交易利润, 毛利润: {trade.pnl:.2f}, 净利润: {trade.pnlcomm:.2f}')
    
    def next(self):
        """策略主逻辑"""
        # 记录投资组合价值
        self.portfolio_values.append(self.broker.getvalue())
        
        # 子类实现具体交易逻辑
        pass
    
    def get_strategy_stats(self) -> Dict:
        """获取策略统计信息"""
        if not self.portfolio_values:
            return {}
        
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        stats = {
            'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
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

class WeightedFactorStrategy(BaseStrategy):
    """策略1: 因子权重加权策略"""
    
    params = (
        ('rsi_weight', 0.25),
        ('macd_weight', 0.25),
        ('bb_weight', 0.20),
        ('ma_weight', 0.20),
        ('momentum_weight', 0.10),
    )
    
    def __init__(self):
        """初始化策略"""
        super().__init__()
        
        # 动量指标
        self.momentum = bt.indicators.MomentumOscillator(self.data.close, period=10)
        
        # 信号权重
        self.signal_weights = {
            'rsi': self.params.rsi_weight,
            'macd': self.params.macd_weight,
            'bb': self.params.bb_weight,
            'ma': self.params.ma_weight,
            'momentum': self.params.momentum_weight
        }
    
    def next(self):
        """策略主逻辑"""
        # 记录投资组合价值
        self.portfolio_values.append(self.broker.getvalue())
        
        # 如果有未完成的订单，等待
        if self.order:
            return
        
        # 计算各因子信号
        signals = self._calculate_factor_signals()
        
        # 计算加权信号
        weighted_signal = self._calculate_weighted_signal(signals)
        
        # 执行交易
        self._execute_trade(weighted_signal)
    
    def _calculate_factor_signals(self) -> Dict[str, float]:
        """计算各因子信号"""
        signals = {}
        
        # RSI信号
        if self.rsi[0] > 70:
            signals['rsi'] = -1  # 超买卖出
        elif self.rsi[0] < 30:
            signals['rsi'] = 1   # 超买卖入
        else:
            signals['rsi'] = 0   # 中性
        
        # MACD信号
        if self.macd.macd[0] > self.macd.signal[0]:
            signals['macd'] = 1  # 金叉买入
        else:
            signals['macd'] = -1 # 死叉卖出
        
        # 布林带信号
        bb_position = (self.data.close[0] - self.bb.lines.bot[0]) / (self.bb.lines.top[0] - self.bb.lines.bot[0])
        if bb_position > 0.8:
            signals['bb'] = -1   # 接近上轨卖出
        elif bb_position < 0.2:
            signals['bb'] = 1    # 接近下轨买入
        else:
            signals['bb'] = 0    # 中性
        
        # 移动平均线信号
        if self.sma_20[0] > self.sma_50[0]:
            signals['ma'] = 1    # 短期均线在长期均线上方，买入
        else:
            signals['ma'] = -1   # 短期均线在长期均线下方，卖出
        
        # 动量信号
        if self.momentum[0] > 0:
            signals['momentum'] = 1   # 正动量买入
        else:
            signals['momentum'] = -1  # 负动量卖出
        
        return signals
    
    def _calculate_weighted_signal(self, signals: Dict[str, float]) -> float:
        """计算加权信号"""
        weighted_sum = 0
        
        for factor, signal in signals.items():
            if factor in self.signal_weights:
                weighted_sum += signal * self.signal_weights[factor]
        
        return weighted_sum
    
    def _execute_trade(self, weighted_signal: float):
        """执行交易"""
        # 获取当前持仓
        position = self.getposition()
        
        if weighted_signal > 0.1 and not position:  # 买入信号且无持仓
            self.log(f'买入信号: {weighted_signal:.3f}')
            self.order = self.buy()
        
        elif weighted_signal < -0.1 and position:  # 卖出信号且有持仓
            self.log(f'卖出信号: {weighted_signal:.3f}')
            self.order = self.sell()

class MaxWeightStrategy(BaseStrategy):
    """策略2: 最大权重信号策略"""
    
    params = (
        ('rsi_weight', 0.25),
        ('macd_weight', 0.25),
        ('bb_weight', 0.20),
        ('ma_weight', 0.20),
        ('momentum_weight', 0.10),
    )
    
    def __init__(self):
        """初始化策略"""
        super().__init__()
        
        # 动量指标
        self.momentum = bt.indicators.MomentumOscillator(self.data.close, period=10)
        
        # 信号权重
        self.signal_weights = {
            'rsi': self.params.rsi_weight,
            'macd': self.params.macd_weight,
            'bb': self.params.bb_weight,
            'ma': self.params.ma_weight,
            'momentum': self.params.momentum_weight
        }
    
    def next(self):
        """策略主逻辑"""
        # 记录投资组合价值
        self.portfolio_values.append(self.broker.getvalue())
        
        # 如果有未完成的订单，等待
        if self.order:
            return
        
        # 计算各因子得分
        factor_scores = self._calculate_factor_scores()
        
        # 选择最大权重信号
        max_weight_signal = self._get_max_weight_signal(factor_scores)
        
        # 执行交易
        self._execute_trade(max_weight_signal)
    
    def _calculate_factor_scores(self) -> Dict[str, float]:
        """计算各因子得分"""
        scores = {}
        
        # RSI得分
        if self.rsi[0] > 70:
            scores['rsi'] = -self.signal_weights['rsi']  # 超买卖出
        elif self.rsi[0] < 30:
            scores['rsi'] = self.signal_weights['rsi']   # 超买卖入
        else:
            scores['rsi'] = 0                            # 中性
        
        # MACD得分
        if self.macd.macd[0] > self.macd.signal[0]:
            scores['macd'] = self.signal_weights['macd']  # 金叉买入
        else:
            scores['macd'] = -self.signal_weights['macd'] # 死叉卖出
        
        # 布林带得分
        bb_position = (self.data.close[0] - self.bb.lines.bot[0]) / (self.bb.lines.top[0] - self.bb.lines.bot[0])
        if bb_position > 0.8:
            scores['bb'] = -self.signal_weights['bb']     # 接近上轨卖出
        elif bb_position < 0.2:
            scores['bb'] = self.signal_weights['bb']      # 接近下轨买入
        else:
            scores['bb'] = 0                              # 中性
        
        # 移动平均线得分
        if self.sma_20[0] > self.sma_50[0]:
            scores['ma'] = self.signal_weights['ma']      # 短期均线在长期均线上方，买入
        else:
            scores['ma'] = -self.signal_weights['ma']     # 短期均线在长期均线下方，卖出
        
        # 动量得分
        if self.momentum[0] > 0:
            scores['momentum'] = self.signal_weights['momentum']   # 正动量买入
        else:
            scores['momentum'] = -self.signal_weights['momentum']  # 负动量卖出
        
        return scores
    
    def _get_max_weight_signal(self, factor_scores: Dict[str, float]) -> Tuple[str, float]:
        """获取最大权重信号"""
        if not factor_scores:
            return None, 0
        
        # 找出得分最高的因子
        max_factor = max(factor_scores.items(), key=lambda x: abs(x[1]))
        factor_name, score = max_factor
        
        return factor_name, score
    
    def _execute_trade(self, max_weight_signal: Tuple[str, float]):
        """执行交易"""
        if max_weight_signal[0] is None:
            return
        
        factor_name, score = max_weight_signal
        
        # 获取当前持仓
        position = self.getposition()
        
        if score > 0.05 and not position:  # 买入信号且无持仓
            self.log(f'最大权重买入信号: {factor_name}, 得分: {score:.3f}')
            self.order = self.buy()
        
        elif score < -0.05 and position:  # 卖出信号且有持仓
            self.log(f'最大权重卖出信号: {factor_name}, 得分: {score:.3f}')
            self.order = self.sell()

class BuyAndHoldStrategy(BaseStrategy):
    """买入持有策略（基准策略）"""
    
    def __init__(self):
        """初始化策略"""
        super().__init__()
        self.bought = False
    
    def next(self):
        """策略主逻辑"""
        # 记录投资组合价值
        self.portfolio_values.append(self.broker.getvalue())
        
        # 如果有未完成的订单，等待
        if self.order:
            return
        
        # 在第一个交易日买入
        if not self.bought:
            self.log('买入持有策略: 买入')
            self.order = self.buy()
            self.bought = True

class BacktraderBacktester:
    """Backtrader回测器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化回测器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.backtest_config = self.config['backtest']
        
    def run_backtest(self, df: pd.DataFrame, strategy_class: bt.Strategy, 
                    strategy_name: str = "Strategy", **kwargs) -> Dict:
        """运行回测
        
        Args:
            df: 数据DataFrame
            strategy_class: 策略类
            strategy_name: 策略名称
            **kwargs: 策略参数
            
        Returns:
            回测结果
        """
        # 创建Cerebro引擎
        cerebro = bt.Cerebro()
        
        # 添加数据
        data = self._prepare_data(df)
        cerebro.adddata(data)
        
        # 设置初始资金
        cerebro.broker.setcash(self.backtest_config['initial_cash'])
        
        # 设置手续费
        cerebro.broker.setcommission(commission=self.backtest_config['commission'])
        
        # 设置滑点
        cerebro.broker.set_slippage_perc(self.backtest_config['slippage'])
        
        # 添加策略
        cerebro.addstrategy(strategy_class, **kwargs)
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # 运行回测
        logger.info(f"开始运行 {strategy_name} 回测...")
        results = cerebro.run()
        
        # 获取策略实例
        strategy = results[0]
        
        # 收集结果
        backtest_results = {
            'strategy_name': strategy_name,
            'final_value': cerebro.broker.getvalue(),
            'total_return': (cerebro.broker.getvalue() - self.backtest_config['initial_cash']) / self.backtest_config['initial_cash'],
            'portfolio_stats': strategy.get_strategy_stats(),
            'analyzers': {
                'sharpe_ratio': results[0].analyzers.sharpe.get_analysis(),
                'drawdown': results[0].analyzers.drawdown.get_analysis(),
                'returns': results[0].analyzers.returns.get_analysis(),
                'trades': results[0].analyzers.trades.get_analysis()
            },
            'trades': strategy.trades,
            'portfolio_values': strategy.portfolio_values
        }
        
        logger.info(f"{strategy_name} 回测完成，最终价值: {backtest_results['final_value']:.2f}")
        
        return backtest_results
    
    def _prepare_data(self, df: pd.DataFrame) -> bt.feeds.PandasData:
        """准备数据格式
        Args:
            df: 数据DataFrame
        Returns:
            Backtrader数据格式
        """
        # 确保日期列格式正确
        df = df.copy()
        if 'date' not in df.columns:
            raise ValueError('数据缺少 date 列')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df_bt = df.set_index('date')
        # 创建Backtrader数据源
        data = bt.feeds.PandasData(
            dataname=df_bt,
            datetime=None,  # 使用索引作为日期
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )
        return data
    
    def compare_strategies(self, df: pd.DataFrame) -> Dict:
        """比较不同策略的性能
        
        Args:
            df: 数据DataFrame
            
        Returns:
            策略比较结果
        """
        strategies = [
            (WeightedFactorStrategy, "WeightedFactor", {}),
            (MaxWeightStrategy, "MaxWeight", {}),
            (BuyAndHoldStrategy, "BuyAndHold", {})
        ]
        
        results = {}
        
        for strategy_class, strategy_name, params in strategies:
            try:
                result = self.run_backtest(df, strategy_class, strategy_name, **params)
                results[strategy_name] = result
            except Exception as e:
                logger.error(f"运行 {strategy_name} 策略失败: {str(e)}")
                results[strategy_name] = None
        
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
        
        # 找出最佳策略（基于夏普比率）
        best_strategy = max(valid_results.items(), 
                          key=lambda x: x[1]['analyzers']['sharpe_ratio'].get('sharperatio', 0))
        
        report['best_strategy'] = best_strategy[0]
        
        # 生成摘要
        for strategy, result in valid_results.items():
            sharpe = result['analyzers']['sharpe_ratio'].get('sharperatio', 0)
            drawdown = result['analyzers']['drawdown'].get('max', {}).get('drawdown', 0)
            
            report['summary'][strategy] = {
                'total_return': f"{result['total_return']:.4f}",
                'sharpe_ratio': f"{sharpe:.4f}",
                'max_drawdown': f"{drawdown:.4f}",
                'final_value': f"{result['final_value']:.2f}",
                'total_trades': len(result['trades'])
            }
        
        return report
    
    def plot_results(self, results: Dict, save_path: str = None):
        """绘制回测结果
        
        Args:
            results: 回测结果
            save_path: 保存路径
        """
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        if save_path is None:
            save_path = "results/backtest_results.png"
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        valid_results = {k: v for k, v in results['results'].items() if v is not None}
        
        if valid_results:
            # 投资组合价值比较
            for strategy_name, result in valid_results.items():
                portfolio_values = result['portfolio_values']
                dates = pd.date_range(start=result['trades'][0]['date'] if result['trades'] else pd.Timestamp.now(),
                                    periods=len(portfolio_values), freq='D')
                axes[0, 0].plot(dates, portfolio_values, label=strategy_name)
            
            axes[0, 0].set_title('投资组合价值比较')
            axes[0, 0].set_ylabel('投资组合价值')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 总收益率比较
            returns = [valid_results[s]['total_return'] for s in valid_results.keys()]
            axes[0, 1].bar(valid_results.keys(), returns)
            axes[0, 1].set_title('总收益率比较')
            axes[0, 1].set_ylabel('总收益率')
            
            # 夏普比率比较
            sharpe_ratios = [valid_results[s]['analyzers']['sharpe_ratio'].get('sharperatio', 0) 
                           for s in valid_results.keys()]
            axes[1, 0].bar(valid_results.keys(), sharpe_ratios)
            axes[1, 0].set_title('夏普比率比较')
            axes[1, 0].set_ylabel('夏普比率')
            
            # 最大回撤比较
            max_drawdowns = [valid_results[s]['analyzers']['drawdown'].get('max', {}).get('drawdown', 0) 
                           for s in valid_results.keys()]
            axes[1, 1].bar(valid_results.keys(), max_drawdowns)
            axes[1, 1].set_title('最大回撤比较')
            axes[1, 1].set_ylabel('最大回撤')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"回测结果图表已保存到: {save_path}")

if __name__ == "__main__":
    # 测试回测策略
    from src.data.data_loader import OilDataLoader
    from src.features.feature_engineering import FeatureEngineer
    
    # 加载数据
    loader = OilDataLoader()
    df = loader.get_data_with_cache(use_cache=True)
    
    # 特征工程
    engineer = FeatureEngineer()
    df_features = engineer.add_technical_indicators(df)
    
    # 创建回测器
    backtester = BacktraderBacktester()
    
    # 运行策略比较
    results = backtester.compare_strategies(df_features)
    
    print("回测完成!")
    print(f"策略比较结果: {results['comparison_report']}")
    
    # 绘制结果
    backtester.plot_results(results) 