import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

# ========== 文件路径（运行前需要修改为你的实际文件路径） ==========
USER_PATH = "D:\大学\课程\数据科学概论\天池竞赛\\fresh_comp_offline\\tianchi_fresh_comp_train_user.csv"   # 用户行为数据
ITEMS_PATH = "D:\大学\课程\数据科学概论\天池竞赛\\fresh_comp_offline\\tianchi_fresh_comp_train_item.csv"  # 商品子集数据
OUTPUT_PATH = "D:\大学\课程\数据科学概论\天池竞赛\\my_result.csv"                     # 输出预测结果

# ========== 读取数据 ==========
print("正在加载数据...")
df = pd.read_csv(USER_PATH, low_memory=False)
items = pd.read_csv(ITEMS_PATH, low_memory=False)

df['time'] = pd.to_datetime(df['time'], errors='coerce')  # 把时间字段转成时间格式

# # ========== 抽样用户（可选，为了加快运行速度） ==========
# unique_users = df['user_id'].nunique()  # 统计总用户数

# sample_size = min(1000, unique_users)   # 最多抽1000个用户
# #sample_size = unique_users

# if sample_size and sample_size < unique_users:
#     sample_users = df['user_id'].drop_duplicates().sample(sample_size, random_state=42).values
#     df_sample = df[df['user_id'].isin(sample_users)].copy()
# else:
#     df_sample = df.copy()

# ========== 不抽样，使用全部用户 ==========
df_sample = df.copy()
label_date = df_sample['time'].dt.date.max()
print(f"使用全部 {df_sample['user_id'].nunique()} 个用户，共 {len(df_sample)} 条行为数据，标签日期为 {label_date}")

label_date = df_sample['time'].dt.date.max()  # 训练数据的最后一天（这里对应12月18日）
print(f"本次使用 {df_sample['user_id'].nunique()} 个用户，共 {len(df_sample)} 条行为数据，标签日期为 {label_date}")


# ========== 商品类别映射 ==========
if 'item_category' in items.columns:
    item_cat_map = dict(items[['item_id','item_category']].drop_duplicates().values)
    df_sample['item_category'] = df_sample['item_id'].map(item_cat_map)
else:
    df_sample['item_category'] = np.nan

# ========== 规则1：基于时间的加权分数 ==========
decay_days = 3.0    # 衰减参数，数值越小，越强调最近行为
fav_weight = 1.5    # 收藏行为比浏览权重更高

# 只取浏览和收藏行为
ev = df_sample[df_sample['behavior_type'].isin([1,2])].copy()
# 距离最后一天（12.18）的天数
ev['days_diff'] = ((pd.to_datetime(label_date) - ev['time']).dt.total_seconds() / (3600*24)).clip(lower=0)
# 距离越近，权重越大（指数衰减）
ev['time_weight'] = np.exp(-ev['days_diff'] / decay_days)
# 收藏行为额外加权
ev['type_weight'] = np.where(ev['behavior_type']==2, fav_weight, 1.0)
# 综合权重
ev['weighted'] = ev['time_weight'] * ev['type_weight']
# 每个用户-商品的加权得分
uw = ev.groupby(['user_id','item_id'], as_index=False)['weighted'].sum().rename(columns={'weighted':'time_weighted_score'})

# ========== 用户-商品基础特征 ==========
df_sample['is_view'] = (df_sample['behavior_type']==1).astype(int)
df_sample['is_fav']  = (df_sample['behavior_type']==2).astype(int)
df_sample['is_cart'] = (df_sample['behavior_type']==3).astype(int)
df_sample['is_buy']  = (df_sample['behavior_type']==4).astype(int)

base = df_sample.groupby(['user_id','item_id'], as_index=False).agg(
    view_cnt = ('is_view','sum'),
    fav_cnt  = ('is_fav','sum'),
    cart_cnt = ('is_cart','sum'),
    buy_cnt  = ('is_buy','sum'),
    last_time = ('time','max')
)

# 合并时间加权分数
feat = base.merge(uw, on=['user_id','item_id'], how='left')
feat['time_weighted_score'] = feat['time_weighted_score'].fillna(0.0)

# 确保商品类别在特征中
feat = feat.merge(items[['item_id','item_category']].drop_duplicates(), on='item_id', how='left')
feat['item_category'] = feat['item_category'].fillna(-1).astype(int)

# ========== 规则2：基于类别购买的调整 ==========
user_cat = df_sample.copy()
# 统计用户在各类别上的购买次数和最后购买时间
buys = user_cat[user_cat['behavior_type']==4].groupby(['user_id','item_category'], as_index=False).agg(
    buy_count=('item_id','count'),
    last_buy_time=('time','max')
)

# 看看用户最后一次购买之后，是否还有浏览/收藏该类别
after = user_cat.merge(buys[['user_id','item_category','last_buy_time']], on=['user_id','item_category'], how='left')
after = after[after['time'] > after['last_buy_time']]
after_interest = after[after['behavior_type'].isin([1,2])].groupby(['user_id','item_category'], as_index=False).size().rename(columns={'size':'after_interest_cnt'})

# 合并购买统计和后续兴趣
cat_adj = buys.merge(after_interest, on=['user_id','item_category'], how='left').fillna(0)

# 规则逻辑：
# - 如果只买过一次，且买完后再也没关注 → 购买意向下降（-1）
# - 如果买过多次，且买完后还持续关注 → 购买意向上升（+2）
# - 其他情况不变（0）
def decide_adj(row):
    if row['buy_count']==1 and row['after_interest_cnt']==0:
        return -1
    if row['buy_count']>=2 and row['after_interest_cnt']>0:
        return 2
    return 0
cat_adj['category_purchase_adjust'] = cat_adj.apply(decide_adj, axis=1)
cat_adj = cat_adj[['user_id','item_category','category_purchase_adjust']]

# 合并类别调整参数
feat = feat.merge(cat_adj, on=['user_id','item_category'], how='left')
feat['category_purchase_adjust'] = feat['category_purchase_adjust'].fillna(0).astype(int)

# ========== 构造标签（12月19日的购买数据） ==========
labels = df_sample[(df_sample['time'].dt.date == label_date) & (df_sample['behavior_type']==4)][['user_id','item_id']].drop_duplicates()
labels['label'] = 1
data = feat.merge(labels, on=['user_id','item_id'], how='left')
data['label'] = data['label'].fillna(0).astype(int)

print("最终数据集维度:", data.shape, "正样本数:", int(data['label'].sum()))

# ========== 训练模型并生成 score ==========
feature_cols = ['view_cnt','fav_cnt','cart_cnt','buy_cnt','time_weighted_score','category_purchase_adjust']
X = data[feature_cols].fillna(0)
y = data['label']

print("开始训练或打分...")
try:
    if y.sum() > 0:
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
        clf.fit(X, y, eval_metric='auc')
        data['score'] = clf.predict_proba(X[feature_cols])[:, 1]
        score_source = "情况 1：使用 LightGBM 模型预测的购买概率"
        print("✅", score_source)
    else:
        raise ValueError("No positives to train on")
except Exception as e:
    score_source = "情况 2：使用规则加权打分法（未训练模型）"
    print("⚙️", score_source, "| 错误信息：", e)
    data['score'] = data['time_weighted_score'] + 0.5 * data['category_purchase_adjust']

# ========== 保存预测结果 ==========
out = data.sort_values(['user_id', 'score'], ascending=[True, False]).groupby('user_id').head(5)[['user_id','item_id','score']]

# 将 score 保留 4 位小数
out['score'] = out['score'].astype(float).round(4)

out.to_csv(OUTPUT_PATH, index=False)
print(f"已保存预测结果到 {OUTPUT_PATH}")
print(f"本次运行中，score 来源：{score_source}")

