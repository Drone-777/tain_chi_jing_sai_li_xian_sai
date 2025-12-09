# recommend_hybrid_fusion.py
# 融合版本：LightGBM滑动窗口框架 + SVM版本的加权特征逻辑
# 目标：利用SVM版的高效人工特征增强LightGBM的预测能力

import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

# ====================== 0. 配置区 ======================
# 路径配置
USER_PATH = r"D:\\大学\\课程\\数据科学概论\\天池竞赛\\fresh_comp_offline\\tianchi_fresh_comp_train_user.csv"
ITEMS_PATH = r"D:\\大学\\课程\\数据科学概论\\天池竞赛\\fresh_comp_offline\\tianchi_fresh_comp_train_item.csv"
OUTPUT_SUBMIT = r"D:\\大学\\课程\\数据科学概论\\天池竞赛\\tianchi_mobile_recommendation_predict1.csv"

# 策略配置
WINDOW_DAYS = 2           # 滑动窗口天数
RANDOM_STATE = 42

# LightGBM 参数 (针对3天窗口优化)
LGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=32,       
    min_child_samples=50, # 稍微调大，防止在小样本上过拟合
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1            # 静默模式，减少警告干扰
)

# 代码 B 中的硬编码权重 (基于全局统计)
# look: ~0.01, like: ~0.65, putin: ~0.39, buy: 1.0
WEIGHTS_B = {
    'look': 20989 / 1863827, # ≈ 0.011
    'like': 20989 / 32506,   # ≈ 0.645
    'putin': 20989 / 53646,  # ≈ 0.391
    'buy': 1.0
}
# =======================================================

def ensure_path_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ================== 1. 读取与预处理 ==================
print("1) 正在读取数据...")
try:
    df = pd.read_csv(USER_PATH, low_memory=False)
    items_subset = pd.read_csv(ITEMS_PATH, low_memory=False)
except FileNotFoundError:
    print("错误：文件未找到，请检查路径配置！")
    sys.exit(1)

# 提取目标商品集合
TARGET_ITEM_SET = set(items_subset['item_id'].astype(str))

# 数据清洗
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['time'])
df['date'] = df['time'].dt.date
df['behavior_type'] = df['behavior_type'].astype(int)
df['user_id'] = df['user_id'].astype(str)
df['item_id'] = df['item_id'].astype(str)

# 映射 item_category
items_subset['item_id'] = items_subset['item_id'].astype(str)
item_cat_map = dict(zip(items_subset['item_id'], items_subset['item_category']))
df['item_category'] = df['item_id'].map(item_cat_map).fillna(-1)

min_date = df['date'].min()
max_date = df['date'].max()
print(f"数据日期范围: {min_date} 到 {max_date}")

# ================== 2. 工具函数 ==================
def dynamic_balance(df_in):
    """动态正负样本平衡"""
    pos = df_in[df_in['label'] == 1]
    neg = df_in[df_in['label'] == 0]
    pos_n = len(pos)
    if pos_n == 0: return df_in
    
    # 动态比例：正样本越少，负样本采样比例越低，避免淹没
    ratio = 5 if pos_n < 100 else (3 if pos_n < 500 else 2)
    
    n_neg = min(len(neg), int(pos_n * ratio))
    neg_sample = neg.sample(n=n_neg, random_state=RANDOM_STATE)
    
    return pd.concat([pos, neg_sample], axis=0).sample(frac=1.0, random_state=RANDOM_STATE)

def search_best_threshold(y_true, y_proba):
    """搜索最佳F1阈值"""
    best_t, best_f1 = 0.5, 0.0
    if y_true.sum() == 0: return 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        f1 = f1_score(y_true, (y_proba >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1

# ================== 3. 特征工程 (融合版) ==================
def build_features_labels(window_start_date, df_all, window_days=WINDOW_DAYS):
    """
    融合了 SVM 版本的特征逻辑：
    1. 引入 weights 加权分
    2. 特别关注窗口最后一天的行为 (模拟 SVM 版只看 12-18 的逻辑)
    """
    ws = pd.to_datetime(window_start_date).date()
    feat_start = ws
    feat_end = ws + timedelta(days=window_days-1) # 窗口最后一天
    label_day = ws + timedelta(days=window_days)  # 预测日

    # 截取窗口内数据
    window_df = df_all[(df_all['date'] >= feat_start) & (df_all['date'] <= feat_end)].copy()
    if window_df.empty: return None

    # --- A. 基础 One-Hot ---
    window_df['is_view'] = (window_df['behavior_type'] == 1).astype(int)
    window_df['is_fav']  = (window_df['behavior_type'] == 2).astype(int)
    window_df['is_cart'] = (window_df['behavior_type'] == 3).astype(int)
    window_df['is_buy']  = (window_df['behavior_type'] == 4).astype(int)

    # --- B. 融合特征 1: 计算 SVM 代码中的加权分 (script_b_score) ---
    # SVM代码公式: look*W1 + like*W2 + putin*W3 + buy*1
    # 我们这里计算每条记录的分数，然后聚合
    window_df['manual_weight'] = 0.0
    window_df.loc[window_df['is_view']==1, 'manual_weight'] = WEIGHTS_B['look']
    window_df.loc[window_df['is_fav']==1, 'manual_weight']  = WEIGHTS_B['like']
    window_df.loc[window_df['is_cart']==1, 'manual_weight'] = WEIGHTS_B['putin']
    window_df.loc[window_df['is_buy']==1, 'manual_weight']  = WEIGHTS_B['buy']

    # --- C. 融合特征 2: 窗口“最后一天”的强特征 ---
    # SVM 代码只看 12-18 (预测日前一天)，我们模拟这个逻辑
    window_df['is_last_day'] = (window_df['date'] == feat_end)
    
    # 聚合
    feat = window_df.groupby(['user_id', 'item_id'], as_index=False).agg(
        # 基础计数
        view_cnt=('is_view', 'sum'),
        fav_cnt=('is_fav', 'sum'),
        cart_cnt=('is_cart', 'sum'),
        buy_cnt=('is_buy', 'sum'),
        # 时间特征
        last_time=('time', 'max'),
        item_category=('item_category', 'first'),
        # [融合] SVM 版加权总分
        script_b_score=('manual_weight', 'sum'),
        # [融合] 最后一天行为计数 (捕捉“昨天加购”的强信号)
        last_day_view_cnt=('is_view', lambda x: x[window_df.loc[x.index, 'is_last_day']].sum()),
        last_day_cart_cnt=('is_cart', lambda x: x[window_df.loc[x.index, 'is_last_day']].sum()),
        last_day_buy_cnt=('is_buy', lambda x: x[window_df.loc[x.index, 'is_last_day']].sum())
    )

    # --- D. 衍生特征 ---
    # 转化率
    feat['cart_ratio'] = feat['cart_cnt'] / (feat['view_cnt'] + 1)
    # 最后交互距离 (小时数)
    feat['hours_to_pred'] = (pd.to_datetime(label_day) - pd.to_datetime(feat['last_time'])).dt.total_seconds() / 3600
    
    # --- E. 构造 Label (如果存在) ---
    label_df = pd.DataFrame()
    if label_day <= df_all['date'].max():
        target = df_all[(df_all['date'] == label_day) & (df_all['behavior_type'] == 4)]
        label_df = target[['user_id', 'item_id']].drop_duplicates()
        label_df['label'] = 1
        data = feat.merge(label_df, on=['user_id', 'item_id'], how='left')
        data['label'] = data['label'].fillna(0).astype(int)
    else:
        data = feat # 预测模式
        # 预测模式下，我们只保留最后一天有过交互的，或者 script_b_score 比较高的
        # 这能大幅减少预测计算量，模拟 SVM 版的筛选逻辑
        data = data[data['script_b_score'] > 0] 

    return data

# ================== 4. 滑动窗口训练 ==================
print(f"2) 开始滑动窗口训练 (Window={WINDOW_DAYS}天)...")

# 验证集截止日期 (使用最后3天作为训练，预测第4天)
validate_end_start = max_date - timedelta(days=WINDOW_DAYS) 
cur_date = min_date

window_f1s = []
window_ths = []
model = None 

# 为了节省时间，我们可以只训练最近的几个窗口 (比如从 12月1日开始)
# 如果你想跑全量，注释掉下面这行
if cur_date < datetime(2014, 12, 1).date():
    cur_date = datetime(2014, 12, 1).date()
    print("   (为加速训练，从 2014-12-01 开始滑动)")

while cur_date <= validate_end_start:
    target_date = cur_date + timedelta(days=WINDOW_DAYS)
    print(f" > Window: {cur_date} -> {target_date - timedelta(days=1)} | Target: {target_date}")
    
    data = build_features_labels(cur_date, df, window_days=WINDOW_DAYS)
    
    if data is None or 'label' not in data.columns or data['label'].sum() < 5:
        cur_date += timedelta(days=1)
        continue

    # 训练
    train_df = dynamic_balance(data)
    cols_drop = ['user_id', 'item_id', 'label', 'time', 'last_time', 'item_category']
    X = train_df.drop(columns=cols_drop, errors='ignore')
    y = train_df['label']

    clf = LGBMClassifier(**LGB_PARAMS)
    clf.fit(X, y)
    model = clf # 保留最后模型

    # 验证
    valid_X = data.drop(columns=cols_drop + ['label'], errors='ignore')
    valid_y = data['label']
    probs = clf.predict_proba(valid_X)[:, 1]
    
    bt, bf1 = search_best_threshold(valid_y, probs)
    window_f1s.append(bf1)
    window_ths.append(bt)
    print(f"   Window F1: {bf1:.4f} (Thresh: {bt:.2f})")
    
    cur_date += timedelta(days=1)

if not window_ths:
    print("训练失败，无有效窗口")
    sys.exit()

# 使用最后几个窗口的阈值平均值
final_thresh = np.mean(window_ths[-3:]) 
print(f"\n训练完成。平均 F1: {np.mean(window_f1s):.4f} | 最终使用阈值: {final_thresh:.3f}")

# ================== 5. 最终预测 (12.19) ==================
print("\n3) 预测 2014-12-19 ...")

# 预测窗口：最后3天 [12.16, 12.17, 12.18]
pred_start = max_date - timedelta(days=WINDOW_DAYS - 1)
print(f"特征区间: {pred_start} 至 {max_date}")

# 构造特征 (此时没有 Label)
pred_data = build_features_labels(pred_start, df, window_days=WINDOW_DAYS)

cols_drop = ['user_id', 'item_id', 'label', 'time', 'last_time', 'item_category']
X_pred = pred_data.drop(columns=cols_drop, errors='ignore')

# 预测
preds = model.predict_proba(X_pred)[:, 1]
pred_data['score'] = preds

# 过滤逻辑
# 1. 阈值过滤
result = pred_data[pred_data['score'] >= final_thresh].copy()

# 2. [关键融合点] 强制包含 SVM 版逻辑：如果是最后一天(12.18)加购的(cart)，且子集中有，必须保留
# 我们给最后一天加购的样本增加额外的“保送”分数，或者直接通过并集保留
last_day_cart_mask = (pred_data['last_day_cart_cnt'] > 0)
cart_rescue = pred_data[last_day_cart_mask].copy()

print(f"模型预测入围数: {len(result)}")
print(f"最后一天加购(SVM逻辑)数: {len(cart_rescue)}")

# 合并两者 (取并集)
final_df = pd.concat([result, cart_rescue]).drop_duplicates(subset=['user_id', 'item_id'])

# 3. 商品子集过滤 (必须做)
final_df = final_df[final_df['item_id'].isin(TARGET_ITEM_SET)]

print(f"最终提交记录数: {len(final_df)}")

# 保存
final_df[['user_id', 'item_id']].to_csv(OUTPUT_SUBMIT, index=False)
print(f"文件已保存: {OUTPUT_SUBMIT}")