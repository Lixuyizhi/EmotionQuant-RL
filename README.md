# EmotionQuant-RL

本项目为基于情绪量化的强化学习交易系统，目前主要针对原油期货进行研究和回测。后续将逐步扩展支持多种不同的期货品种。

## 项目简介

- 利用情绪数据与多因子特征，结合强化学习方法，进行期货量化交易策略的开发与回测。
- 当前已实现原油期货的相关数据处理、特征工程、策略训练与回测流程。
- 项目结构清晰，便于扩展至其他期货品种。

## 主要功能

- 数据处理与特征工程
- 强化学习模型训练
- 策略回测与评估
- 支持自定义配置与参数调整
- 独立数据集回测系统
- 多数据源支持（本地Excel文件和akshare）
- 多模型支持（PPO、A2C等算法）

## 未来计划

- 支持更多期货品种（如黄金、螺纹钢等）
- 丰富情绪因子与特征库
- 增加多种强化学习算法与策略模板

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

### 3. 数据源切换工具
项目提供了便捷的数据源切换工具：

```bash
# 查看当前数据源配置
python scripts/switch_data_source.py show

# 切换到本地数据源
python scripts/switch_data_source.py local sc2210_major_contracts_2017_30min.xlsx

# 切换到akshare数据源
python scripts/switch_data_source.py akshare
```

### 4. 运行程序
```bash
# 运行主程序
python main.py

# 运行数据使用示例
python example_data_usage.py

# 可参考 notebooks/example_usage.ipynb 进行交互式实验
```

### 5. 回测系统使用

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

回测结果保存在 `backtest_results/` 目录下，包含模型信息、数据信息、回测统计和性能指标。

#### 最佳实践
1. **数据分割策略**: 训练(70%) + 验证(15%) + 测试(15%) + 回测(独立时间段)
2. **时间选择**: 选择与训练数据不重叠的时间段进行回测
3. **多模型比较**: 对同一时间段测试不同模型的性能

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
├── data/                      # 数据存放目录
├── logs/                      # 日志文件目录
├── models/                    # 训练好的模型存放目录
├── results/                   # 回测与实验结果目录
├── scripts/                   # 辅助脚本目录
├── notebooks/                 # Jupyter Notebook 示例
└── src/                       # 核心源代码
    ├── __init__.py
    ├── backtest/              # 回测相关代码
    ├── data/                  # 数据处理相关代码
    ├── features/              # 特征工程相关代码
    │   ├── __init__.py
    │   └── feature_engineering.py
    ├── models/                # 模型相关代码
    └── strategies/            # 策略相关代码
        ├── __init__.py
        └── backtrader_strategies.py
```

如需详细介绍每个目录和文件的作用，也可以告诉我！

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

### 调试模式

启用详细日志：
```bash
export PYTHONPATH=.
python -u backtest_integration.py --model models/PPO_SignalWeightTradingEnv_model.zip
```

## 联系方式

如有问题或建议，欢迎联系项目维护者。
