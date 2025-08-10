# 因子权重可视化脚本使用说明

## 概述

本目录包含了用于分析和可视化强化学习环境中因子权重变化的脚本。这些脚本可以帮助你理解模型如何学习分配不同因子的权重，以及权重随时间的变化模式。

## 脚本说明

### 1. `visualize_weights.py` - 基础可视化脚本

**功能**: 提供基础的权重时间序列和统计分析可视化功能

**主要函数**:
- `visualize_weights_timeseries()`: 生成时间序列折线图和热力图
- `visualize_weights_statistics()`: 生成统计分析图表（柱状图、箱线图、相关性热力图）
- `create_sample_weights_data()`: 创建示例权重数据用于测试

**使用方法**:
```bash
cd scripts
python visualize_weights.py
```

### 2. `extract_weights_from_env.py` - 权重数据提取和分析脚本

**功能**: 从训练好的强化学习环境中提取权重数据，进行深度分析

**主要函数**:
- `extract_weights_from_env()`: 从环境中提取权重历史记录
- `analyze_weights_dynamics()`: 分析权重动态变化特征
- `save_weights_analysis()`: 保存分析结果到文件

**使用方法**:
```bash
cd scripts
python extract_weights_from_env.py
```

## 权重数据结构

### 权重历史记录格式

权重历史记录 `weights_history` 是一个列表，每个元素是一个numpy数组：

```python
weights_history = [
    np.array([0.3, 0.2, 0.3, 0.2]),  # 第1步的权重
    np.array([0.4, 0.1, 0.3, 0.2]),  # 第2步的权重
    np.array([0.2, 0.4, 0.2, 0.2]),  # 第3步的权重
    # ... 更多时间步
]
```

每个权重数组包含对应时间步各个因子的权重值，权重总和为1。

### 信号因子名称

默认的信号因子包括：
- `RSI_signal`: RSI指标信号
- `BB_signal`: 布林带信号
- `SMA_signal`: 移动平均信号
- `MACD_signal`: MACD信号

## 可视化输出

### 1. 时间序列图表 (`weights_timeseries.png`)

包含两个子图：
- **折线图**: 显示每个因子权重随时间的变化趋势
- **热力图**: 用颜色深浅表示权重大小，直观显示权重分布

### 2. 统计分析图表 (`weights_statistics.png`)

包含四个子图：
- **平均权重对比**: 各因子的平均权重及标准差
- **权重范围对比**: 各因子的权重变化范围
- **权重分布箱线图**: 各因子权重的分布特征
- **权重相关性热力图**: 因子间权重的相关性分析

## 分析结果

### 保存的文件

1. **`weights_history.csv`**: 权重历史数据的表格格式
2. **`weights_history.npy`**: 权重历史数据的numpy格式
3. **`weights_analysis.json`**: 详细的权重分析结果

### 分析内容

#### 基础统计信息
- 平均权重、标准差、最小值、最大值、中位数

#### 趋势分析
- 权重变化趋势（递增/递减）
- 趋势强度（相关系数）
- 趋势斜率

#### 波动性分析
- 整体波动性
- 波动性变化趋势
- 最大波动幅度

#### 相关性分析
- 因子间权重的相关系数矩阵
- 平均相关性水平

#### 主导性分析
- 各因子成为主导因子的时间比例
- 连续主导时间
- 主导时的平均权重

## 实际使用场景

### 1. 训练过程监控

在强化学习训练过程中，定期提取权重数据，观察模型学习进展：
```python
# 在训练循环中
if episode % 100 == 0:
    weights_history = env.weights_history
    # 保存或可视化权重数据
```

### 2. 模型性能分析

训练完成后，分析权重变化模式，理解模型决策逻辑：
```python
# 训练完成后
analysis = analyze_weights_dynamics(env.weights_history, env.enabled_signals)
print("模型学习到的权重模式:", analysis)
```

### 3. 策略优化

基于权重分析结果，优化因子配置或调整训练参数：
```python
# 分析哪些因子权重较低
low_weight_factors = [name for name, stats in analysis['summary_stats'].items() 
                     if stats['mean'] < 0.1]
print("权重较低的因子:", low_weight_factors)
```

## 环境差异说明

### MaxWeightTradingEnv vs SignalWeightTradingEnv

两个环境的核心区别在于交易决策机制：

- **MaxWeightTradingEnv**: 赢者通吃策略，只使用权重最大的因子信号
- **SignalWeightTradingEnv**: 综合评估策略，考虑所有信号的加权贡献

但权重记录格式完全相同，都可以使用这些脚本进行分析。

## 依赖要求

```bash
pip install numpy pandas matplotlib seaborn
```

## 注意事项

1. **数据格式**: 确保权重历史记录是numpy数组的列表
2. **权重归一化**: 权重应该已经归一化，总和为1
3. **时间步数量**: 建议至少1000个时间步以获得有意义的分析结果
4. **内存使用**: 大量时间步数据可能占用较多内存

## 扩展功能

可以根据需要添加更多分析功能：
- 权重变化的周期性分析
- 市场状态与权重的关系分析
- 权重预测模型
- 实时权重监控仪表板
