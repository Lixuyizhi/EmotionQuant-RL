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

## 未来计划

- 支持更多期货品种（如黄金、螺纹钢等）
- 丰富情绪因子与特征库
- 增加多种强化学习算法与策略模板

## 使用方法

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 运行主程序：
   ```bash
   python main.py
   ```
3. 可参考 `notebooks/example_usage.ipynb` 进行交互式实验。

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

## 联系方式

如有问题或建议，欢迎联系项目维护者。 