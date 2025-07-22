import os
import re
import yaml
import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, A2C, DQN

from src.models.signal_weight_env import SignalWeightTradingEnv
from src.models.max_weight_env import MaxWeightTradingEnv

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. 加载配置
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 2. 加载测试数据（你可以根据实际情况修改）
from src.data.data_loader import OilDataLoader
from src.features.feature_engineering import FeatureEngineer

loader = OilDataLoader()
df = loader.get_data_with_cache(use_cache=True)
engineer = FeatureEngineer()
df_features = engineer.add_technical_indicators(df)
df_features = engineer.add_target_variables(df_features)
df_features = engineer.create_trading_features(df_features)

# 数据分割，取测试集
train_split = config['model_training']['training']['train_split']
val_split = config['model_training']['training']['validation_split']
test_split = config['model_training']['training']['test_split']
n = len(df_features)
train_end = int(n * train_split)
val_end = train_end + int(n * val_split)
test_df = df_features.iloc[val_end:].copy()

# 3. 遍历模型文件
model_dir = "models"
model_files = [f for f in os.listdir(model_dir) if f.endswith("_model.zip")]

# 4. 定义环境映射
env_map = {
    "SignalWeightTradingEnv": SignalWeightTradingEnv,
    "MaxWeightTradingEnv": MaxWeightTradingEnv
}
algo_map = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN
}

results = {}

for model_file in model_files:
    # 解析算法和环境名
    match = re.match(r"(\w+)_(\w+)_model\.zip", model_file)
    if not match:
        continue
    algo, env_name = match.group(1), match.group(2)
    logger.info(f"正在评估模型: {model_file} (算法: {algo}, 环境: {env_name})")
    # 加载环境
    env_cls = env_map.get(env_name)
    if env_cls is None:
        logger.warning(f"未知环境类型: {env_name}")
        continue
    test_env = env_cls(test_df, "config/config.yaml")
    # 加载模型
    model_path = os.path.join(model_dir, model_file)
    model_cls = algo_map.get(algo)
    if model_cls is None:
        logger.warning(f"未知算法类型: {algo}")
        continue
    model = model_cls.load(model_path)
    # 评估
    obs = test_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    done = False
    total_reward = 0
    step_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step_result = test_env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, done, truncated, info = step_result
            done = done or truncated
        total_reward += reward
        step_count += 1
    stats = test_env.get_portfolio_stats()
    results[model_file] = {
        "algorithm": algo,
        "env": env_name,
        "total_reward": float(total_reward),
        "step_count": step_count,
        "portfolio_stats": stats
    }
    logger.info(f"评估完成: {model_file} -> {stats}")

# 5. 输出对比结果
print("\n模型对比结果：")
for model_file, res in results.items():
    stats = res["portfolio_stats"]
    print(f"{model_file}:")
    print(f"  算法: {res['algorithm']}, 环境: {res['env']}")
    print(f"  总收益率: {stats.get('total_return', 0):.4f}")
    print(f"  夏普比率: {stats.get('sharpe_ratio', 0):.4f}")
    print(f"  最大回撤: {stats.get('max_drawdown', 0):.4f}")
    print(f"  总交易次数: {stats.get('total_trades', 0)}")
    print("-" * 40) 