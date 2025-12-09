# recommend_optimized_3days.py
# 针对天池移动推荐算法比赛优化
# 策略：3天滑动窗口 + 自动阈值 + 强制商品子集过滤

import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from lightgbm import LGBMClassifier

# ====================== 配置区 ======================
# 请确保路径正确
USER_PATH = r"D:\\大学\\课程\\数据科学概论\\天池竞赛\\fresh_comp_offline\\tianchi_fresh_comp_train_user.csv"
ITEMS_PATH = r"D:\\大学\\课程\\数据科学概论\\天池竞赛\\fresh_comp_offline\\tianchi_fresh_comp_train_item.csv"
OUTPUT_SUBMIT = r"D:\\大学\\课程\\数据科学概论\\天池竞赛\\tianchi_mobile_recommendation_predict.csv" # 题目要求的标准文件名

# 核心修改：使用前3天预测第4天
WINDOW_DAYS = 3           
BASE_NEG_POS_RATIO = 2    # 负正样本比例
RANDOM_STATE = 42

LGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=32,        # 窗口变小，特征变少，防止过拟合减小叶子节点
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
# ====================================================

def ensure_path_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ================== 读取与预处理 ==================
print("1) 正在读取数据...")
# 读取用户行为数据 D
df = pd.read_csv(USER_PATH, low_memory=False)
# 读取商品子集 P (题目要求的预测范围)
items_subset = pd.read_csv(ITEMS_PATH, low_memory=False)

# 提取商品子集的 item_id 集合，用于最后过滤
TARGET_ITEM_SET = set(items_subset['item_id'].astype(str))

# 数据清洗
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['time'])
df['date'] = df['time'].dt.date
df['behavior_type'] = df['behavior_type'].astype(int)
df['user_id'] = df['user_id'].astype(str)
df['item_id'] = df['item_id'].astype(str)

# 映射 item_category
if 'item_category' in items_subset.columns:
    # 注意：这里只映射了子集中的 category，全集中的可能缺失，但这不影响，因为我们只关心子集
    item_cat_map = dict(items_subset[['item_id','item_category']].drop_duplicates().values)
    # 将 item_id 转为 str 以匹配
    items_subset['item_id'] = items_subset['item_id'].astype(str)
    item_cat_map = dict(zip(items_subset['item_id'], items_subset['item_category']))
    df['item_category'] = df['item_id'].map(item_cat_map).fillna(-1)
else:
    df['item_category'] = -1

min_date = df['date'].min()
max_date = df['date'].max()
print(f"数据日期范围: {min_date} 到 {max_date}")

# 行为权重
BEHAVIOR_WEIGHT = {1:1.0, 2:2.0, 3:3.0, 4:5.0}

# ================== 动态样本平衡 ==================
def dynamic_balance(df_in, base_ratio=BASE_NEG_POS_RATIO):
    pos = df_in[df_in['label'] == 1]
    neg = df_in[df_in['label'] == 0]
    pos_n = len(pos)
    
    if pos_n == 0: return df_in
    
    # 动态调整采样比例
    if pos_n < 100: ratio = 5
    elif pos_n < 500: ratio = 3
    else: ratio = base_ratio
        
    n_neg = min(len(neg), int(pos_n * ratio))
    neg_sample = neg.sample(n=n_neg, random_state=RANDOM_STATE)
    
    balanced = pd.concat([pos, neg_sample], axis=0)
    balanced = balanced.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return balanced

# ================== 阈值搜索 ==================
def search_best_threshold(y_true, y_proba, thresholds=np.arange(0.1, 0.91, 0.05)):
    best_t, best_f1 = 0.5, 0.0
    if y_true.sum() == 0: return 0.5, 0.0
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1

# ================== 特征工程 (3天窗口版) ==================
def build_features_labels(window_start_date, df_all, window_days=WINDOW_DAYS):
    """
    用 [start, start + window_days - 1] 的数据构造特征
    如果 mode='train': 标签是 start + window_days 当天的购买
    如果 mode='predict': 没有标签，只返回特征
    """
    ws = pd.to_datetime(window_start_date).date()
    feat_start = ws
    feat_end = ws + timedelta(days=window_days-1)
    label_day = ws + timedelta(days=window_days) # 第4天

    # 截取窗口内数据
    window_df = df_all[(df_all['date'] >= feat_start) & (df_all['date'] <= feat_end)].copy()
    if window_df.empty: return None, None

    # --- 1. 基础聚合特征 ---
    window_df['is_view'] = (window_df['behavior_type'] == 1).astype(int)
    window_df['is_fav']  = (window_df['behavior_type'] == 2).astype(int)
    window_df['is_cart'] = (window_df['behavior_type'] == 3).astype(int)
    window_df['is_buy']  = (window_df['behavior_type'] == 4).astype(int)

    base = window_df.groupby(['user_id', 'item_id'], as_index=False).agg(
        view_cnt=('is_view', 'sum'),
        fav_cnt=('is_fav', 'sum'),
        cart_cnt=('is_cart', 'sum'),
        buy_cnt=('is_buy', 'sum'),
        last_time=('time', 'max'),
        item_category=('item_category', 'first')
    )

    # --- 2. 时间衰减加权特征 (针对3天短窗口优化衰减系数) ---
    ev = window_df.copy()
    # 计算距离特征截止日期的天数差 (0, 1, 2)
    ev['days_diff'] = ((pd.to_datetime(feat_end) - ev['time']).dt.total_seconds() / (3600*24)).clip(lower=0)
    # 3天窗口比较短，衰减系数设小一点，让最近的一天权重更高
    ev['time_weight'] = np.exp(-ev['days_diff'] / 1.0) 
    ev['type_weight'] = ev['behavior_type'].map(BEHAVIOR_WEIGHT).fillna(1.0)
    ev['weighted'] = ev['time_weight'] * ev['type_weight']
    
    uw = ev.groupby(['user_id', 'item_id'], as_index=False)['weighted'].sum().rename(columns={'weighted': 'score_weighted'})
    feat = base.merge(uw, on=['user_id', 'item_id'], how='left').fillna(0)

    # --- 3. 转化率特征 (User Conversion) ---
    feat['buy_ratio'] = feat['buy_cnt'] / (feat['view_cnt'] + 1)
    feat['cart_ratio'] = feat['cart_cnt'] / (feat['view_cnt'] + 1)
    
    # --- 4. 距离最后一次交互的时间间隔 ---
    feat['last_gap_hours'] = feat['last_time'].apply(
        lambda x: (pd.to_datetime(feat_end) + timedelta(days=1) - pd.to_datetime(x)).total_seconds() / 3600
    )

    # --- 获取标签 (仅针对 label_day 当天的购买) ---
    # 只有当 label_day 在数据集中存在时才构造标签
    label_df = pd.DataFrame()
    if label_day <= df_all['date'].max():
        target_buys = df_all[(df_all['date'] == label_day) & (df_all['behavior_type'] == 4)]
        # 去重，一个人买多次同一个商品算一次 positive
        label_df = target_buys[['user_id', 'item_id']].drop_duplicates()
        label_df['label'] = 1
        
        # 合并标签
        dataL = feat.merge(label_df, on=['user_id', 'item_id'], how='left')
        dataL['label'] = dataL['label'].fillna(0).astype(int)
    else:
        # 预测模式，没有 label
        dataL = feat
        dataL['label'] = 0

    return dataL

# ================== 核心逻辑：滑动窗口验证 ==================
print(f"2) 开始滑动窗口训练 (窗口大小: {WINDOW_DAYS}天)...")

# 这里的 max_date 是 2014-12-18
# 最后一个能验证的窗口：特征(12.15-12.17) -> 标签(12.18)
# 所以 start 循环到 12.15 即可
validate_end_start = max_date - timedelta(days=WINDOW_DAYS) 

cur_date = min_date
window_f1s = []
window_ths = []

model = None # 存储最后一个模型

while cur_date <= validate_end_start:
    # 构造特征：[cur, cur+2] -> 预测 [cur+3]
    print(f" > 正在处理窗口: {cur_date} 至 {cur_date + timedelta(days=WINDOW_DAYS-1)} | 预测目标: {cur_date + timedelta(days=WINDOW_DAYS)}")
    
    dataL = build_features_labels(cur_date, df, window_days=WINDOW_DAYS)
    
    if dataL is None or len(dataL) == 0:
        cur_date += timedelta(days=1)
        continue
        
    # 平衡样本
    balanced = dynamic_balance(dataL)
    
    # 准备训练集
    drop_cols = ['user_id', 'item_id', 'label', 'time', 'last_time', 'item_category']
    feat_cols = [c for c in balanced.columns if c not in drop_cols]
    
    X = balanced[feat_cols]
    y = balanced['label']
    
    if y.sum() < 5: # 正样本太少，跳过
        print("   (跳过：正样本不足)")
        cur_date += timedelta(days=1)
        continue

    # 训练模型
    clf = LGBMClassifier(**LGB_PARAMS)
    clf.fit(X, y)
    model = clf # 更新最新模型
    
    # 在当前窗口的全量数据上验证 (不进行欠采样)
    X_all = dataL[feat_cols]
    y_all = dataL['label']
    y_proba = clf.predict_proba(X_all)[:, 1]
    
    # 搜索最佳阈值 (Optimize)
    best_t, best_f1 = search_best_threshold(y_all, y_proba)
    
    window_f1s.append(best_f1)
    window_ths.append(best_t)
    
    print(f"   F1-Score: {best_f1:.5f} (Threshold: {best_t:.2f})")
    
    cur_date += timedelta(days=1)

if not window_f1s:
    print("错误：没有生成有效的训练窗口。")
    sys.exit(1)

avg_f1 = np.mean(window_f1s)
# 使用中位数作为全局推荐阈值
final_threshold = float(np.median(window_ths))
print(f"\n训练结束。平均 F1: {avg_f1:.5f} | 推荐阈值: {final_threshold:.3f}")

# ================== 最终预测 (12.19) ==================
print("\n3) 生成最终预测结果 (Target: 2014-12-19)...")

# 预测窗口：特征使用最后3天 [12.16, 12.17, 12.18]
pred_start_date = max_date - timedelta(days=WINDOW_DAYS - 1) # 12.18 - 2 = 12.16
print(f"预测特征区间: {pred_start_date} 至 {max_date}")

# 1. 构造特征
pred_data = build_features_labels(pred_start_date, df, window_days=WINDOW_DAYS)

# 2. 准备预测数据
drop_cols = ['user_id', 'item_id', 'label', 'time', 'last_time', 'item_category']
feat_cols = [c for c in pred_data.columns if c not in drop_cols]
X_pred = pred_data[feat_cols]

# 3. 使用最后一个窗口训练的模型(或者你也可以把所有数据合并重训一次)进行预测
# 这里使用最后一个滑动窗口产生的模型，通常已经包含了最近的趋势
y_pred_proba = model.predict_proba(X_pred)[:, 1]

pred_data['score'] = y_pred_proba

# 4. 筛选与生成结果
# 过滤阈值
result = pred_data[pred_data['score'] >= final_threshold].copy()

# ================== 关键步骤：过滤商品子集 ==================
# 题目要求：只预测 Subset P 中的商品
# 如果不加这一步，F1分数会极低，因为预测了大量不在考察范围内的商品
print(f"过滤前记录数: {len(result)}")
result = result[result['item_id'].isin(TARGET_ITEM_SET)]
print(f"过滤商品子集(P)后记录数: {len(result)}")

# 5. 去重并保存
# 题目要求：tianchi_mobile_recommendation_predict.csv，包含 user_id, item_id
final_submission = result[['user_id', 'item_id']].drop_duplicates()

ensure_path_dir(OUTPUT_SUBMIT)
final_submission.to_csv(OUTPUT_SUBMIT, index=False)

print(f"\n✅ 预测完成！文件已保存至: {OUTPUT_SUBMIT}")
print("Good Luck!")