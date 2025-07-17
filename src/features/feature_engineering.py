import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Dict, Optional, Tuple
import logging
import yaml

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化特征工程器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 读取新的配置结构
        data_config = self.config.get('data_processing', {})
        features_config = data_config.get('features', {})
        
        self.technical_indicators = features_config.get('technical_indicators', ['SMA', 'RSI', 'MACD', 'BB'])
        self.lookback_periods = features_config.get('lookback_periods', [20])
        self.feature_scaling = features_config.get('feature_scaling', 'standard')
        
        # 初始化缩放器
        self.scaler = self._get_scaler()
        
    def _get_scaler(self):
        """获取特征缩放器"""
        if self.feature_scaling == "standard":
            return StandardScaler()
        elif self.feature_scaling == "minmax":
            return MinMaxScaler()
        elif self.feature_scaling == "robust":
            return RobustScaler()
        else:
            return StandardScaler()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加技术指标后的DataFrame
        """
        df = df.copy()
        
        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info("开始计算技术指标...")
        
        # 移动平均线
        if "SMA" in self.technical_indicators:
            for period in self.lookback_periods:
                df[f'SMA_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
        
        if "EMA" in self.technical_indicators:
            for period in self.lookback_periods:
                df[f'EMA_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
        
        # RSI
        if "RSI" in self.technical_indicators:
            df['RSI_14'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        if "MACD" in self.technical_indicators:
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_histogram'] = macd.macd_diff()
        
        # 布林带
        if "BB" in self.technical_indicators:
            bb = ta.volatility.BollingerBands(df['close'])
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_middle'] = bb.bollinger_mavg()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_width'] = bb.bollinger_wband()
            df['BB_percent'] = bb.bollinger_pband()
        
        # ATR
        if "ATR" in self.technical_indicators:
            for period in [14, 21]:
                df[f'ATR_{period}'] = ta.volatility.average_true_range(
                    df['high'], df['low'], df['close'], window=period
                )
        
        # OBV
        if "OBV" in self.technical_indicators:
            df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # CCI
        if "CCI" in self.technical_indicators:
            for period in [14, 21]:
                df[f'CCI_{period}'] = ta.trend.cci(
                    df['high'], df['low'], df['close'], window=period
                )
        
        # Williams %R
        if "Williams_R" in self.technical_indicators:
            for period in [14, 21]:
                df[f'Williams_R_{period}'] = ta.momentum.williams_r(
                    df['high'], df['low'], df['close'], lbp=period
                )
        
        # Stochastic
        if "Stochastic" in self.technical_indicators:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
        
        logger.info(f"技术指标计算完成，特征数量: {len(df.columns) - 6}")  # 减去原始6列
        
        return df
    

    
    def add_target_variables(self, df: pd.DataFrame, 
                           target_periods: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
        """添加目标变量（未来收益率）
        
        Args:
            df: 数据DataFrame
            target_periods: 目标预测周期列表
            
        Returns:
            添加目标变量后的DataFrame
        """
        df = df.copy()
        
        for period in target_periods:
            df[f'target_return_{period}d'] = df['close'].shift(-period) / df['close'] - 1
        
        # 添加分类目标变量
        for period in target_periods:
            returns = df[f'target_return_{period}d']
            df[f'target_signal_{period}d'] = np.where(returns > 0.01, 1,  # 买入信号
                                                     np.where(returns < -0.01, -1, 0))  # 卖出信号
        
        return df
    
    def create_trading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交易特征
        
        Args:
            df: 数据DataFrame
            
        Returns:
            包含交易特征的DataFrame
        """
        df = df.copy()
        
        # 计算各种交易信号
        df = self._add_trading_signals(df)
        
        # 填充所有_signal结尾的列的NaN为0
        signal_cols = [col for col in df.columns if col.endswith('_signal')]
        df[signal_cols] = df[signal_cols].fillna(0)
        
        return df
    
    def _add_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加交易信号"""
        # RSI信号
        if 'RSI_14' in df.columns:
            df['RSI_signal'] = np.where(df['RSI_14'] > 70, -1,  # 超买卖出
                                      np.where(df['RSI_14'] < 30, 1, 0))  # 超买卖入
        
        # MACD信号
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            df['MACD_signal'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
        
        # 布林带信号
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            df['BB_signal'] = np.where(df['close'] > df['BB_upper'], -1,  # 突破上轨卖出
                                     np.where(df['close'] < df['BB_lower'], 1, 0))  # 突破下轨买入
        
        # 移动平均线信号
        if 'SMA_20' in df.columns:
            df['SMA_signal'] = np.where(df['close'] > df['SMA_20'], 1, -1)  # 价格在均线上方买入
        
        return df
    

    
    def prepare_features_for_ml(self, df: pd.DataFrame, 
                              target_col: str = 'target_return_1d') -> Tuple[pd.DataFrame, pd.Series]:
        """准备机器学习特征
        
        Args:
            df: 数据DataFrame
            target_col: 目标变量列名
            
        Returns:
            特征DataFrame和目标变量Series
        """
        # 选择特征列
        feature_columns = [col for col in df.columns 
                          if col not in ['date', 'open', 'high', 'low', 'close', 'volume'] 
                          and not col.startswith('target_')]
        
        # 删除包含缺失值的行
        df_clean = df.dropna(subset=feature_columns + [target_col])
        
        X = df_clean[feature_columns]
        y = df_clean[target_col]
        
        logger.info(f"特征矩阵形状: {X.shape}")
        logger.info(f"目标变量形状: {y.shape}")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """缩放特征
        
        Args:
            X_train: 训练特征
            X_test: 测试特征
            
        Returns:
            缩放后的特征
        """
        # 拟合缩放器
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    


if __name__ == "__main__":
    # 测试特征工程
    from src.data.data_loader import OilDataLoader
    
    # 加载数据
    loader = OilDataLoader()
    df = loader.get_data_with_cache(use_cache=True)
    
    # 特征工程
    engineer = FeatureEngineer()
    df_features = engineer.add_technical_indicators(df)
    df_features = engineer.add_target_variables(df_features)
    df_features = engineer.create_trading_features(df_features)
    
    # 准备ML特征
    X, y = engineer.prepare_features_for_ml(df_features)
    
    print("特征工程完成!")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    print(f"特征列名: {list(X.columns[:10])}...")  # 显示前10个特征 