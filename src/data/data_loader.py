import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import yaml
from typing import Optional, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OilDataLoader:
    """中国原油期货数据加载器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化数据加载器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 读取新的配置结构
        data_config = self.config.get('data_processing', {})
        data_source = data_config.get('data_source', {})
        
        self.data_path = data_source.get('data_path', 'data/')
        self.symbol = data_source.get('symbol', 'SC')
        self.start_date = data_source.get('start_date', '2020-01-01')
        self.end_date = data_source.get('end_date', '2024-01-01')
        
        # 确保数据目录存在
        os.makedirs(self.data_path, exist_ok=True)
        
    def get_oil_futures_data(self, symbol: str = None, start_date: str = None, 
                           end_date: str = None) -> pd.DataFrame:
        """获取中国原油期货数据
        
        Args:
            symbol: 期货合约代码，默认为SC（原油期货主力）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        symbol = symbol or self.symbol
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        
        logger.info(f"正在获取 {symbol} 期货数据，时间范围: {start_date} 到 {end_date}")
        
        try:
            # 使用akshare获取中国原油期货主力合约数据
            if symbol == "SC":
                # 获取原油期货主力合约数据
                df = ak.futures_main_sina(symbol="SC0")
            else:
                # 获取指定合约数据
                df = ak.futures_zh_daily_sina(symbol=symbol)
            
            # 重命名列（根据实际返回的列数调整）
            if len(df.columns) == 6:
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            elif len(df.columns) == 8:
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'hold', 'settle']
            else:
                # 如果列数不匹配，只取前6列
                df = df.iloc[:, :6]
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            
            # 过滤日期范围
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # 按日期排序
            df = df.sort_values('date').reset_index(drop=True)
            
            # 转换数据类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除缺失值
            df = df.dropna()
            
            logger.info(f"成功获取 {len(df)} 条数据记录")
            
            return df
            
        except Exception as e:
            logger.error(f"获取数据失败: {str(e)}")
            raise
    
    def get_multiple_contracts_data(self, symbols: list = None) -> dict:
        """获取多个合约的数据
        
        Args:
            symbols: 合约代码列表
            
        Returns:
            包含多个合约数据的字典
        """
        symbols = symbols or ["SC0", "SC1", "SC2"]  # 主力、次主力、远月合约
        
        data_dict = {}
        for symbol in symbols:
            try:
                data = self.get_oil_futures_data(symbol=symbol)
                data_dict[symbol] = data
                logger.info(f"成功获取 {symbol} 合约数据")
            except Exception as e:
                logger.warning(f"获取 {symbol} 合约数据失败: {str(e)}")
                continue
        
        return data_dict
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """保存数据到文件
        
        Args:
            df: 数据DataFrame
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"{self.symbol}_{self.start_date}_{self.end_date}.csv"
        
        filepath = os.path.join(self.data_path, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"数据已保存到: {filepath}")
        
        return filepath
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """从文件加载数据
        
        Args:
            filename: 文件名
            
        Returns:
            数据DataFrame
        """
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"从 {filepath} 加载了 {len(df)} 条数据记录")
        
        return df
    
    def get_data_with_cache(self, use_cache: bool = True) -> pd.DataFrame:
        """获取数据（支持缓存）
        
        Args:
            use_cache: 是否使用缓存
            
        Returns:
            数据DataFrame
        """
        cache_filename = f"{self.symbol}_{self.start_date}_{self.end_date}.csv"
        cache_filepath = os.path.join(self.data_path, cache_filename)
        
        if use_cache and os.path.exists(cache_filepath):
            logger.info("使用缓存数据")
            return self.load_data(cache_filename)
        
        # 获取新数据
        df = self.get_oil_futures_data()
        
        # 保存到缓存
        if use_cache:
            self.save_data(df, cache_filename)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据质量
        
        Args:
            df: 数据DataFrame
            
        Returns:
            数据是否有效
        """
        # 检查必要的列
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error("数据缺少必要的列")
            return False
        
        # 检查数据类型
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            logger.error("日期列格式不正确")
            return False
        
        # 检查数值范围
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                logger.error(f"{col} 列包含非正数值")
                return False
        
        # 检查OHLC逻辑
        if not ((df['high'] >= df['low']).all() and 
                (df['high'] >= df['open']).all() and 
                (df['high'] >= df['close']).all() and
                (df['low'] <= df['open']).all() and 
                (df['low'] <= df['close']).all()):
            logger.error("OHLC数据逻辑错误")
            return False
        
        # 检查缺失值
        if df.isnull().any().any():
            logger.warning("数据包含缺失值")
        
        logger.info("数据验证通过")
        return True

if __name__ == "__main__":
    # 测试数据加载器
    loader = OilDataLoader()
    
    # 获取数据
    df = loader.get_data_with_cache(use_cache=True)
    
    # 验证数据
    if loader.validate_data(df):
        print("数据获取和验证成功!")
        print(f"数据形状: {df.shape}")
        print(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
        print("\n前5行数据:")
        print(df.head())
    else:
        print("数据验证失败!") 