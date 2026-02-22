import pandas as pd

# 读取PRD格网元数据
input_file = 'data/grid_metadata/PRD_grid_metadata.csv'
output_file = 'data/grid_metadata/shenzhen_grid.csv'

print(f"读取文件: {input_file}")
df = pd.read_csv(input_file)

print(f"总格网数量: {len(df)}")
print(f"城市列表: {df['city_name'].unique().tolist()}")

# 提取深圳市的格网
shenzhen_df = df[df['city_name'] == '深圳市'].copy()

print(f"\n深圳市格网数量: {len(shenzhen_df)}")
print(f"深圳市区县列表: {shenzhen_df['area_name'].unique().tolist()}")

# 保存到新文件
shenzhen_df.to_csv(output_file, index=False)

print(f"\n已保存到: {output_file}")
print(f"输出文件前5行:")
print(shenzhen_df.head())
