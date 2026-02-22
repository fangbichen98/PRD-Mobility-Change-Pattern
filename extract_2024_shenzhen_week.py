import pandas as pd
import numpy as np
from datetime import datetime
import os

# 读取深圳市格网ID
print("读取深圳市格网数据...")
shenzhen_grid = pd.read_csv('data/grid_metadata/shenzhen_grid.csv')
shenzhen_grid_ids = set(shenzhen_grid['grid_id'].values)
print(f"深圳市格网数量: {len(shenzhen_grid_ids)}")

# 定义处理函数
def extract_shenzhen_week(year, input_file, output_file):
    print(f"\n{'='*60}")
    print(f"处理 {year} 年数据")
    print(f"{'='*60}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")

    # 首先检查日期范围以确定第一周
    print("\n检查日期范围...")
    date_chunks = pd.read_csv(input_file, usecols=['date_dt'], chunksize=1000000)
    all_dates = set()
    for chunk in date_chunks:
        all_dates.update(chunk['date_dt'].unique())
        if len(all_dates) > 10:  # 只需要前几个日期
            break

    sorted_dates = sorted(all_dates)
    print(f"发现日期: {sorted_dates[:10]}")

    # 确定第一周的日期范围（取前7个唯一日期）
    first_week_dates = sorted(sorted_dates)[:7]
    print(f"第一周日期范围: {first_week_dates[0]} 到 {first_week_dates[-1]}")
    first_week_set = set(first_week_dates)

    # 处理数据并筛选
    print(f"\n开始提取数据...")
    chunk_size = 1000000
    chunks_processed = 0
    total_rows = 0
    kept_rows = 0

    # 使用 iterator 模式读取大文件
    reader = pd.read_csv(input_file, chunksize=chunk_size)

    first_chunk = True
    for chunk in reader:
        chunks_processed += 1
        total_rows += len(chunk)

        if chunks_processed % 10 == 0:
            print(f"已处理 {chunks_processed} 个chunks, 共 {total_rows:,} 行...")

        # 筛选条件
        # 1. 日期在第一周内
        mask_date = chunk['date_dt'].isin(first_week_set)

        # 2. origin和destination都在深圳
        mask_o = chunk['o_grid_500'].isin(shenzhen_grid_ids)
        mask_d = chunk['d_grid_500'].isin(shenzhen_grid_ids)

        # 应用筛选
        filtered_chunk = chunk[mask_date & mask_o & mask_d]

        if len(filtered_chunk) > 0:
            kept_rows += len(filtered_chunk)
            if first_chunk:
                # 第一个有数据的chunk，写入文件头
                filtered_chunk.to_csv(output_file, index=False, mode='w')
                first_chunk = False
            else:
                # 追加数据，不写表头
                filtered_chunk.to_csv(output_file, index=False, mode='a', header=False)

        if chunks_processed % 50 == 0:
            print(f"  - 已保留 {kept_rows:,} 行")

    print(f"\n完成!")
    print(f"总处理行数: {total_rows:,}")
    print(f"保留行数: {kept_rows:,}")
    print(f"保留比例: {kept_rows/total_rows*100:.2f}%")

    # 验证输出文件
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"输出文件大小: {file_size:.2f} MB")

        # 读取并显示前几行
        result_df = pd.read_csv(output_file, nrows=5)
        print(f"\n输出文件前5行:")
        print(result_df)

        # 统计信息
        full_result = pd.read_csv(output_file)
        print(f"\n统计信息:")
        print(f"  - 总记录数: {len(full_result):,}")
        print(f"  - 日期分布:")
        print(full_result['date_dt'].value_counts().sort_index())
        print(f"  - 独特origin格网数: {full_result['o_grid_500'].nunique()}")
        print(f"  - 独特destination格网数: {full_result['d_grid_500'].nunique()}")
        print(f"  - 总流量: {full_result['num_total'].sum():,}")

# 只处理2024年
extract_shenzhen_week(
    year=2024,
    input_file='data/2024.csv',
    output_file='data/2024_shenzhen_week.csv'
)

print(f"\n{'='*60}")
print(f"完成!")
print(f"{'='*60}")
