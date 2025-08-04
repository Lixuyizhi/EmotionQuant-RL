# EmotionQuant-RL

本项目为基于情绪量化的强化学习交易系统，目前主要针对原油期货进行研究和回测。后续将逐步扩展支持多种不同的期货品种。

## 项目简介

- 利用情绪数据与多因子特征，结合强化学习方法，进行期货量化交易策略的开发与回测。
- 当前已实现原油期货的相关数据处理、特征工程、策略训练与回测流程。
- 项目结构清晰，便于扩展至其他期货品种。
- **支持动态信号数量配置**：可在配置文件中灵活选择启用的交易信号，自动调整动作空间和观察空间。

## 主要功能

- 数据处理与特征工程
- 强化学习模型训练
- 策略回测与评估
- 支持自定义配置与参数调整
- 独立数据集回测系统
- 多数据源支持（本地Excel文件和akshare）
- 多模型支持（PPO、A2C等算法）
- **动态信号支持**：支持任意数量的交易信号组合
- **智能参数优化**：自动调整交易阈值、成本控制等参数
- **增强的回测系统**：改进的结果文件命名和详细性能分析

## 最新更新

### v2.0 主要改进
- ✅ **动态信号支持**：完全支持配置文件中的 `enabled_signals` 参数
- ✅ **参数优化**：优化交易阈值、手续费、滑点等关键参数
- ✅ **回测结果改进**：文件名包含算法名、环境名和模型名
- ✅ **性能分析增强**：详细的交易统计和信号权重分析

### 环境支持
- **SignalWeightTradingEnv**：信号权重交易环境，支持动态信号数量
- **MaxWeightTradingEnv**：最大权重交易环境，支持动态信号数量

## 未来计划

- 支持更多期货品种（如黄金、螺纹钢等）
- 丰富情绪因子与特征库
- 增加多种强化学习算法与策略模板
- 实时交易接口集成
- 多因子模型优化

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 数据源配置
项目支持两种数据源：

#### 本地数据源
- 将Excel数据文件放在 `data/` 目录下
- 修改 `config/config.yaml` 中的配置：
  ```yaml
  data_processing:
    data_source:
      source: local
      local_file: your_data_file.xlsx
  ```

#### akshare数据源
- 使用akshare实时获取数据
- 修改 `config/config.yaml` 中的配置：
  ```yaml
  data_processing:
    data_source:
      source: akshare
  ```

### 3. 交易信号配置
在 `config/config.yaml` 中可以灵活配置启用的交易信号：

```yaml
trading_signals:
  enabled_signals: ['RSI_signal', 'BB_signal', 'SMA_signal', 'MACD_signal']
  weight_min: 0.0
  weight_max: 1.0
```

支持任意数量的信号组合，系统会自动调整动作空间和观察空间。

### 4. 环境参数优化
关键参数说明：

```yaml
signal_weight_env:
  # 交易阈值（影响交易频率）
  buy_threshold: 0.1      # 买入信号阈值
  sell_threshold: -0.1    # 卖出信号阈值
  
  # 仓位管理
  position_size: 0.08     # 每次交易仓位比例
  max_position_ratio: 0.8 # 最大仓位比例
  
  # 成本控制
  transaction_fee: 0.0003 # 交易手续费
  slippage: 0.00005      # 滑点成本
  
  # 奖励机制
  reward_scale: 0.8       # 奖励缩放因子
  risk_penalty: 0.001     # 风险惩罚系数
```

### 5. 数据源切换工具
项目提供了便捷的数据源切换工具：

```bash
# 查看当前数据源配置
python scripts/switch_data_source.py show

# 切换到本地数据源
python scripts/switch_data_source.py local sc2210_major_contracts_2017_30min.xlsx

# 切换到akshare数据源
python scripts/switch_data_source.py akshare
```

### 6. 模型训练
```bash
# 训练信号权重环境模型
python model_training.py --env signal_weight_env --algorithm PPO

# 训练最大权重环境模型
python model_training.py --env max_weight_env --algorithm PPO

# 使用A2C算法训练
python model_training.py --env signal_weight_env --algorithm A2C
```

### 7. 回测系统使用

#### 基本回测命令
```bash
# 使用指定时间范围回测
python backtest_integration.py --model models/PPO_SignalWeightTradingEnv_model.zip --start_date 2024-01-01 --end_date 2024-06-30

# 使用akshare数据源回测
python backtest_integration.py --model models/A2C_SignalWeightTradingEnv_model.zip --data_source akshare --start_date 2023-01-01 --end_date 2023-12-31

# 不保存结果（仅显示）
python backtest_integration.py --model models/PPO_MaxWeightTradingEnv_model.zip --no_save

# 运行完整回测示例
python run_backtest_example.py
```

#### 回测参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型文件路径 | 配置文件中的model_path |
| `--start_date` | 回测开始日期 (YYYY-MM-DD) | 配置文件中的start_date |
| `--end_date` | 回测结束日期 (YYYY-MM-DD) | 配置文件中的end_date |
| `--data_source` | 数据源类型 (local/akshare) | local |
| `--config` | 配置文件路径 | config/config.yaml |
| `--no_save` | 不保存回测结果 | False |

#### 回测结果解读
关键性能指标包括：
- **总收益率 (total_return)**: 模型在回测期间的总收益
- **夏普比率 (sharpe_ratio)**: 风险调整后的收益指标
- **最大回撤 (max_drawdown)**: 最大损失幅度
- **波动率 (volatility)**: 收益的波动程度
- **交易次数 (total_trades)**: 总交易次数
- **信号权重分析**: 各交易信号的权重分布和使用情况

#### 回测结果文件命名
回测结果文件现在包含完整的模型信息：
```
PPO_SignalWeightTradingEnv_best_model_backtest_20250804_144938.json
```
格式：`{算法名}_{环境名}_{模型名}_backtest_{时间戳}.json`

回测结果保存在 `backtest_results/` 目录下，包含：
- 模型信息（算法、环境、训练时间等）
- 数据信息（时间范围、数据量等）
- 回测统计（收益率、夏普比率等）
- 性能指标（交易次数、平均奖励等）
- 信号权重分析（各信号的使用情况）

#### 最佳实践
1. **数据分割策略**: 训练(70%) + 验证(15%) + 测试(15%) + 回测(独立时间段)
2. **时间选择**: 选择与训练数据不重叠的时间段进行回测
3. **多模型比较**: 对同一时间段测试不同模型的性能
4. **参数调优**: 根据回测结果调整交易阈值和成本参数
5. **信号组合**: 尝试不同的信号组合，找到最优配置

## 目录结构

```
EmotionQuant-RL/
├── README.md                  # 项目说明文档
├── requirements.txt           # Python依赖包列表
├── main.py                    # 主程序入口
├── run_demo.py                # 示例运行脚本
├── backtest_integration.py    # 回测集成模块
├── model_training.py          # 模型训练主模块
├── data_processing.py         # 数据处理模块
├── test_config_reading.py     # 配置读取测试
├── test_signal_weight_env.py  # 信号权重环境测试
├── test_system.py             # 系统测试
├── .gitignore                 # Git忽略文件
├── .DS_Store                  # macOS系统文件
│
├── config/                    # 配置文件目录
│   └── config.yaml           # 主配置文件
├── data/                      # 数据存放目录
├── logs/                      # 日志文件目录
├── models/                    # 训练好的模型存放目录
│   ├── best/                 # 最佳模型目录
│   ├── training_results/     # 训练结果
│   └── checkpoints/          # 检查点文件
├── backtest_results/          # 回测结果目录
├── results/                   # 其他实验结果目录
├── scripts/                   # 辅助脚本目录
│   ├── adjust_config.py      # 配置调整工具
│   ├── compare_trained_models.py # 模型比较工具
│   ├── rl_backtrader_test.py # 回测测试
│   └── switch_data_source.py # 数据源切换工具
├── notebooks/                 # Jupyter Notebook 示例
└── src/                       # 核心源代码
    ├── __init__.py
    ├── backtest/              # 回测相关代码
    ├── data/                  # 数据处理相关代码
    │   ├── __init__.py
    │   └── data_loader.py
    ├── features/              # 特征工程相关代码
    │   ├── __init__.py
    │   └── feature_engineering.py
    ├── models/                # 模型相关代码
    │   ├── __init__.py
    │   ├── max_weight_env.py  # 最大权重交易环境
    │   ├── signal_weight_env.py # 信号权重交易环境
    │   └── rl_trainer.py      # 强化学习训练器
    └── strategies/            # 策略相关代码
        ├── __init__.py
        └── backtrader_strategies.py
```

## 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   FileNotFoundError: 模型文件不存在: models/xxx_model.zip
   ```
   解决：检查模型文件路径是否正确

2. **数据加载失败**
   ```
   ValueError: 数据验证失败
   ```
   解决：检查数据文件格式和完整性

3. **环境创建失败**
   ```
   ModuleNotFoundError: No module named 'xxx_env'
   ```
   解决：检查环境模块是否正确安装

4. **动态信号配置问题**
   ```
   ValueError: 动作空间形状不匹配
   ```
   解决：确保配置文件中的 `enabled_signals` 与训练时一致

5. **回测结果异常**
   ```
   总奖励为负但总收益率为正
   ```
   说明：这是正常现象，奖励基于交易成本，收益率基于资产价值变化

### 调试模式

启用详细日志：
```bash
export PYTHONPATH=.
python -u backtest_integration.py --model models/PPO_SignalWeightTradingEnv_model.zip
```

### 性能优化建议

1. **减少过度交易**：
   - 提高交易阈值（buy_threshold, sell_threshold）
   - 增加最小交易金额（min_trade_amount）
   - 降低仓位大小（position_size）

2. **降低交易成本**：
   - 减少交易手续费（transaction_fee）
   - 降低滑点成本（slippage）
   - 优化风险惩罚系数（risk_penalty）

3. **提高模型稳定性**：
   - 调整奖励缩放因子（reward_scale）
   - 优化信号权重范围（weight_min, weight_max）
   - 平衡交易频率和收益

## 联系方式

如有问题或建议，欢迎联系项目维护者。
