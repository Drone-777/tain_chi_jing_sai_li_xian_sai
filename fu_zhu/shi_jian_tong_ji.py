import pandas as pd

# 1. 读取文件
file_path = "D:\大学\课程\数据科学概论\天池竞赛\\fresh_comp_offline\\tianchi_fresh_comp_train_user.csv"
df = pd.read_csv(file_path)

# 2. 转换时间列为 datetime 类型（格式到小时）
df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H")

# 3. 提取小时信息
df["hour"] = df["time"].dt.hour

# 4. 按小时分组统计整个月份每个小时的数据量
hourly_counts = df["hour"].value_counts().sort_index()

# 5. 保存结果为 CSV 文件
hourly_counts.to_csv("D:\Program\class_code\Python\\tian_chi\\fu_zhu\\hourly_aggregate_count.csv", header=["count"])

# 6. 打印结果
print("各小时段数据统计（单位：条）\n")
print(hourly_counts)
print(f"\n统计完成，共 {len(hourly_counts)} 个小时的数据。结果已保存为 hourly_aggregate_count.csv。")
