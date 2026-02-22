import pandas as pd

print("="*80)
print("深圳市第一周格网间流量数据提取完成")
print("="*80)

# 读取2021年数据
print("\n【2021年数据统计】")
df_2021 = pd.read_csv('data/2021_shenzhen_week.csv')
print(f"文件: 2021_shenzhen_week.csv (1.1 GB)")
print(f"日期范围: 2021-03-15 至 2021-03-21 (7天)")
print(f"总记录数: {len(df_2021):,}")
print(f"总流量: {df_2021['num_total'].sum():,}")
print(f"平均流量: {df_2021['num_total'].mean():.2f}")
print(f"独特格网数: {len(set(df_2021['o_grid_500']).union(set(df_2021['d_grid_500']))):,}")
print(f"\n各天记录数:")
for date, count in df_2021['date_dt'].value_counts().sort_index().items():
    print(f"  {date}: {count:,}")

# 读取2024年数据
print("\n【2024年数据统计】")
df_2024 = pd.read_csv('data/2024_shenzhen_week.csv')
print(f"文件: 2024_shenzhen_week.csv (862 MB)")
print(f"日期范围: 2024-03-18 至 2024-03-24 (7天)")
print(f"总记录数: {len(df_2024):,}")
print(f"总流量: {df_2024['num_total'].sum():,.0f}")
print(f"平均流量: {df_2024['num_total'].mean():.2f}")
print(f"独特格网数: {len(set(df_2024['o_grid_500']).union(set(df_2024['d_grid_500']))):,}")
print(f"\n各天记录数:")
for date, count in df_2024['date_dt'].value_counts().sort_index().items():
    print(f"  {date}: {count:,}")

# 对比分析
print("\n【两年对比分析】")
record_change = (len(df_2024) - len(df_2021)) / len(df_2021) * 100
flow_change = (df_2024['num_total'].sum() - df_2021['num_total'].sum()) / df_2021['num_total'].sum() * 100

print(f"记录数变化: {record_change:+.2f}%")
print(f"  2021: {len(df_2021):,} 条")
print(f"  2024: {len(df_2024):,} 条")
print(f"\n总流量变化: {flow_change:+.2f}%")
print(f"  2021: {df_2021['num_total'].sum():,}")
print(f"  2024: {df_2024['num_total'].sum():,.0f}")

print("\n"+"="*80)
print("数据提取成功！文件已保存到 data/ 目录")
print("="*80)
