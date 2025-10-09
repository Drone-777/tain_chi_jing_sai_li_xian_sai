import pandas as pd

df = pd.read_csv("D:\大学\课程\数据科学概论\天池竞赛\my_result.csv")
print(df['score'].describe(percentiles=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99]))
