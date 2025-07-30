#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据源切换脚本
快速切换本地数据源和akshare数据源
"""

import yaml
import os
import sys
from pathlib import Path

def switch_data_source(source_type: str, local_file: str = None):
    """切换数据源
    
    Args:
        source_type: 数据源类型 ('local' 或 'akshare')
        local_file: 本地文件名 (当source_type为'local'时使用)
    """
    config_path = Path("config/config.yaml")
    
    if not config_path.exists():
        print("❌ 配置文件不存在: config/config.yaml")
        return False
    
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改数据源配置
    data_source = config['data_processing']['data_source']
    data_source['source'] = source_type
    
    if source_type == 'local' and local_file:
        data_source['local_file'] = local_file
        print(f"✅ 已切换到本地数据源: {local_file}")
    elif source_type == 'akshare':
        # 移除本地文件配置
        if 'local_file' in data_source:
            del data_source['local_file']
        print("✅ 已切换到akshare数据源")
    else:
        print("❌ 无效的数据源类型或缺少本地文件名")
        return False
    
    # 保存配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"📝 配置文件已更新: {config_path}")
    return True

def show_current_config():
    """显示当前配置"""
    config_path = Path("config/config.yaml")
    
    if not config_path.exists():
        print("❌ 配置文件不存在")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_source = config['data_processing']['data_source']
    current_source = data_source.get('source', 'unknown')
    local_file = data_source.get('local_file', 'None')
    
    print("📊 当前数据源配置:")
    print(f"   数据源类型: {current_source}")
    if current_source == 'local':
        print(f"   本地文件: {local_file}")
        # 检查文件是否存在
        file_path = Path(f"data/{local_file}")
        if file_path.exists():
            print(f"   ✅ 文件存在: {file_path}")
        else:
            print(f"   ❌ 文件不存在: {file_path}")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("📖 数据源切换工具")
        print("=" * 40)
        print("使用方法:")
        print("  python scripts/switch_data_source.py show")
        print("  python scripts/switch_data_source.py local <filename>")
        print("  python scripts/switch_data_source.py akshare")
        print()
        print("示例:")
        print("  python scripts/switch_data_source.py show")
        print("  python scripts/switch_data_source.py local sc2210_major_contracts_2017_30min.xlsx")
        print("  python scripts/switch_data_source.py akshare")
        print()
        show_current_config()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'show':
        show_current_config()
    elif command == 'local':
        if len(sys.argv) < 3:
            print("❌ 请指定本地文件名")
            print("示例: python scripts/switch_data_source.py local sc2210_major_contracts_2017_30min.xlsx")
            return
        local_file = sys.argv[2]
        switch_data_source('local', local_file)
    elif command == 'akshare':
        switch_data_source('akshare')
    else:
        print(f"❌ 未知命令: {command}")
        print("可用命令: show, local, akshare")

if __name__ == "__main__":
    main() 