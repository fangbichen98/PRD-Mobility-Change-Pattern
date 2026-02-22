import pandas as pd
import os

def analyze_file(filepath, year):
    print(f"\n{'='*70}")
    print(f"{year} 年深圳市第一周流量数据分析")
    print(f"{'='*70}")

    # 读取数据（只加载需要的列以节省内存）
    print(f"读取文件: {filepath}")
    df = pd.read_csv(filepath)

    file_size_mb = os.path.getsize(filepath) / (1024**2)
    print(f"\n基本统计:")
    print(f"  总记录数: {len(df):,}")
    print(f"  文件大小: {file_size_mb:.2f} MB")

    print(f"\n日期分布:")
    date_counts = df['date_dt'].value_counts().sort_index()
    for date, count in date_counts.items():
        print(f"  {date}: {count:,} 条记录")

    print(f"\n格网统计:")
    unique_origins = df['o_grid_500'].nunique()
    unique_destinations = df['d_grid_500'].nunique()
    all_grids = set(df['o_grid_500']).union(set(df['d_grid_500']))
    print(f"  独特origin格网数: {unique_origins:,}")
    print(f"  独特destination格网数: {unique_destinations:,}")
    print(f"  总计独特格网数: {len(all_grids):,}")

    print(f"\n流量统计:")
    total_flow = df['num_total'].sum()
    avg_flow = df['num_total'].mean()
    max_flow = df['num_total'].max()
    print(f"  总流量: {total_flow:,}")
    print(f"  平均流量: {avg_flow:.2f}")
    print(f"  最大单次流量: {max_flow:,}")

    # 检查是否有非深圳格网
    shenzhen_grid = pd.read_csv('data/grid_metadata/shenzhen_grid.csv')
    shenzhen_grid_ids = set(shenzhen_grid['grid_id'].values)

    non_shenzhen_o = set(df['o_grid_500']) - shenzhen_grid_ids
    non_shenzhen_d = set(df['d_grid_500']) - shenzhen_grid_ids

    if non_shenzhen_o:
        print(f"\n⚠️  警告: 发现 {len(non_shenzhen_o)} 个非深圳origin格网")
    if non_shenzhen_d:
        print(f"\n⚠️  警告: 发现 {len(non_shenzhen_d)} 个非深圳destination格网")

    if not non_shenzhen_o and not non_shenzhen_d:
        print(f"\n✓ 所有格网均为深圳市格网")

    print(f"\n前5条记录:")
    print(df.head())

    return df

# 分析2021年数据
df_2021 = analyze_file('data/2021_shenzhen_week.csv', 2021)

# 分析2024年数据
df_2024 = analyze_file('data/2024_shenzhen_week.csv', 2024)

print(f"\n{'='*70}")
print(f"两年对比分析")
print(f"{'='*70}")
print(f"2021年记录数: {len(df_2021):,}")
print(f"2024年记录数: {len(df_2024):,}")
print(f"变化: {len(df_2024) - len(df_2021):,} ({(len(df_2024)/len(df_2021) - 1)*100:+.2f}%)")
print(f"2021年总流量: {df_2021['num_total'].sum():,}")
print(f"2024年总流量: {df_2024['num_total'].sum():,}")
print(f"变化: {df_2024['num_total'].sum() - df_2021['num_total'].sum():,} ({(df_2024['num_total'].sum()/df_2021['num_total'].sum() - 1)*100:+.2f}%)")
