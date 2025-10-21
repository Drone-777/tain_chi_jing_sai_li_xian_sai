# recommend_with_balance.py
# 说明：滑动窗口 + 样本平衡 + 保留你原来的 4 条规则（时间加权、类别购买调整、意向调整、全局偏好）
# 最后使用最后6天训练模型并预测12月19日购买，并输出提交文件（user_id,item_id）

import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# ========== 配置区（运行前请修改以下路径） ==========
USER_PATH = r"D:\\大学\\课程\\数据科学概论\\天池竞赛\\fresh_comp_offline\\tianchi_fresh_comp_train_user.csv"
ITEMS_PATH = r"D:\\大学\\课程\\数据科学概论\\天池竞赛\\fresh_comp_offline\\tianchi_fresh_comp_train_item.csv"
OUTPUT_SUBMIT = r"D:\\大学\\课程\\数据科学概论\\天池竞赛\\recommend_20141219_submit.csv"    # 最终提交（不含score）
OUTPUT_WITH_SCORE = r"D:\\大学\\课程\\数据科学概论\\天池竞赛\\recommend_20141219_with_score.csv"  # 含score备查

# ========== 全局参数（可调） ==========
WINDOW_DAYS = 6       # 用前 6 天预测第 7 天
NEG_POS_RATIO = 2     # 负样本数 = 正样本数 * NEG_POS_RATIO（欠采样比）
THRESHOLD_FINAL = 0.35  # 最后的置信度阈值（可用交叉验证自动调整）
RANDOM_STATE = 42

# LightGBM 默认参数（可调整）
LGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=8
)

# ========== 工具函数 ==========
def ensure_path_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ========== 读取并预处理数据 ==========
print("1) 读取数据（可能较慢）...")
df = pd.read_csv(USER_PATH, low_memory=False)
items = pd.read_csv(ITEMS_PATH, low_memory=False)

# 时间字段转换
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['time'])  # 去掉有问题的行（若有）
df['date'] = df['time'].dt.date

# 合并 item 的 category 信息（如果 items 表包含 item_category）
if 'item_category' in items.columns:
    item_cat_map = dict(items[['item_id','item_category']].drop_duplicates().values)
    df['item_category'] = df['item_id'].map(item_cat_map)
else:
    df['item_category'] = -1

# 为了稳定，确保 behavior_type 为 int
df['behavior_type'] = df['behavior_type'].astype(int)

# 全局可查看的日期范围
min_date = df['date'].min()
max_date = df['date'].max()
print(f"数据日期范围: {min_date} 到 {max_date}")

# ========== 规则：类型权重（与之前规则1相同思路） ==========
# 浏览=1, 收藏=2, 加购物车=3, 购买=4
BEHAVIOR_WEIGHT = {1: 1.0, 2: 1.5, 3: 2.0, 4: 4.0}

# ========== 构建单窗口（前6天 -> 第7天）的特征与标签函数 ==========
def build_features_labels(window_start_date, df_all, window_days=WINDOW_DAYS):
    """
    window_start_date: datetime.date，窗口起始日（第1天）
    使用 window_start_date ... window_start_date+window_days-1 作为特征（前6天）
    目标/标签为 window_start_date+window_days （第7天） 的购买行为（behavior_type==4）
    返回： feat_df (user_id,item_id,... features)、 labels_df (user_id,item_id,label)
    """
    ws = pd.to_datetime(window_start_date).date()
    feat_start = ws
    feat_end = ws + timedelta(days=window_days-1)
    label_day = ws + timedelta(days=window_days)

    # 取特征区间的数据
    window_df = df_all[(df_all['date'] >= feat_start) & (df_all['date'] <= feat_end)].copy()
    if window_df.empty:
        return None, None

    # 取第7天的购买行为作为标签
    label_df = df_all[(df_all['date'] == label_day) & (df_all['behavior_type'] == 4)][['user_id','item_id']].drop_duplicates()
    label_df['label'] = 1

    # --- 特征构造：基于你原来的 4 条规则 ---
    # 基础计数：每个 user-item 在窗口内各行为计数
    window_df['is_view'] = (window_df['behavior_type']==1).astype(int)
    window_df['is_fav']  = (window_df['behavior_type']==2).astype(int)
    window_df['is_cart'] = (window_df['behavior_type']==3).astype(int)
    window_df['is_buy']  = (window_df['behavior_type']==4).astype(int)

        # ============== 修复后的 base 聚合（包含 item_category） ==============
    # 在聚合 user-item 基础统计时把 item_category 一并保留（用 first）
    base = window_df.groupby(['user_id','item_id'], as_index=False).agg(
        view_cnt = ('is_view','sum'),
        fav_cnt  = ('is_fav','sum'),
        cart_cnt = ('is_cart','sum'),
        buy_cnt  = ('is_buy','sum'),
        last_time = ('time','max'),
        item_category = ('item_category','first')   # <-- 保留类别信息，避免后续 merge 报错
    )

    # 规则1 时间加权得分（包含浏览/收藏/加购/购买）
    ev = window_df[window_df['behavior_type'].isin([1,2,3,4])].copy()
    ev['days_diff'] = ((pd.to_datetime(feat_end) - ev['time']).dt.total_seconds() / (3600*24)).clip(lower=0)
    decay_days = 3.0
    ev['time_weight'] = np.exp(-ev['days_diff'] / decay_days)
    ev['type_weight'] = ev['behavior_type'].map(BEHAVIOR_WEIGHT).fillna(1.0)
    ev['weighted'] = ev['time_weight'] * ev['type_weight']
    uw = ev.groupby(['user_id','item_id'], as_index=False)['weighted'].sum().rename(columns={'weighted':'time_weighted_score'})

    # 合并基础统计与时间加权分数（此时 feat 会包含 item_category）
    feat = base.merge(uw, on=['user_id','item_id'], how='left')
    feat['time_weighted_score'] = feat['time_weighted_score'].fillna(0.0)

    # 规则2：基于类别购买的调整（使用窗口内的购买行为）
    user_cat = window_df.copy()
    buys = user_cat[user_cat['behavior_type']==4].groupby(['user_id','item_category'], as_index=False).agg(
        buy_count=('item_id','count'),
        last_buy_time=('time','max')
    )
    after = user_cat.merge(buys[['user_id','item_category','last_buy_time']], on=['user_id','item_category'], how='left')
    after = after[after['time'] > after['last_buy_time']]
    after_interest = after[after['behavior_type'].isin([1,2])].groupby(['user_id','item_category'], as_index=False).size().rename(columns={'size':'after_interest_cnt'})
    cat_adj = buys.merge(after_interest, on=['user_id','item_category'], how='left').fillna(0)
    def decide_adj(row):
        if row['buy_count']==1 and row['after_interest_cnt']==0:
            return -1
        if row['buy_count']>=2 and row['after_interest_cnt']>0:
            return 2
        return 0
    if not cat_adj.empty:
        cat_adj['category_purchase_adjust'] = cat_adj.apply(decide_adj, axis=1)
        cat_adj = cat_adj[['user_id','item_category','category_purchase_adjust']]
    else:
        cat_adj = pd.DataFrame(columns=['user_id','item_category','category_purchase_adjust'])

    feat = feat.merge(cat_adj, on=['user_id','item_category'], how='left')
    feat['category_purchase_adjust'] = feat['category_purchase_adjust'].fillna(0).astype(int)

    # 规则3：基于用户收藏/加购转化率的购买意向修正（用窗口内用户统计）
    user_fav = window_df[window_df['behavior_type'] == 2].groupby('user_id')['item_id'].nunique().rename('fav_total')
    user_fav_buy = window_df[window_df['behavior_type'] == 4].groupby('user_id')['item_id'].nunique().rename('buy_total')
    user_fav_merge = pd.concat([user_fav, user_fav_buy], axis=1).fillna(0)
    user_fav_merge['fav_buy_ratio'] = np.where(user_fav_merge['fav_total'] > 0,
                                               user_fav_merge['buy_total'] / user_fav_merge['fav_total'], 0)
    user_cart = window_df[window_df['behavior_type'] == 3].groupby('user_id')['item_id'].nunique().rename('cart_total')
    user_cart_buy = window_df[window_df['behavior_type'] == 4].groupby('user_id')['item_id'].nunique().rename('buy_total2')
    user_cart_merge = pd.concat([user_cart, user_cart_buy], axis=1).fillna(0)
    user_cart_merge['cart_buy_ratio'] = np.where(user_cart_merge['cart_total'] > 0,
                                                 user_cart_merge['buy_total2'] / user_cart_merge['cart_total'], 0)
    intent = pd.concat([user_fav_merge['fav_buy_ratio'], user_cart_merge['cart_buy_ratio']], axis=1).fillna(0)
    intent['intent_score'] = 0.4 * intent['fav_buy_ratio'] + 0.6 * intent['cart_buy_ratio']
    def intent_adjust(x):
        if x > 0.7:
            return 0.2
        elif x < 0.3:
            return -0.2
        else:
            return 0.0
    intent['intent_adjust'] = intent['intent_score'].apply(intent_adjust)
    feat = feat.merge(intent['intent_adjust'], left_on='user_id', right_index=True, how='left')
    feat['intent_adjust'] = feat['intent_adjust'].fillna(0.0)

    # 规则4：全局类别偏好（使用整个窗口内的购买分布）
    global_pref = (window_df[window_df['behavior_type'] == 4]
                   .groupby('item_category')['item_id']
                   .count()
                   .rename('global_buy_count'))
    if not global_pref.empty:
        global_pref = (global_pref / global_pref.max()).rename('global_pref_score')
        global_pref = global_pref.reset_index()
        feat = feat.merge(global_pref, on='item_category', how='left')
        feat['global_pref_score'] = feat['global_pref_score'].fillna(0.0)
    else:
        feat['global_pref_score'] = 0.0

    # 额外衍生特征（增强区分性）
    feat['buy_ratio'] = feat['buy_cnt'] / (feat['view_cnt'] + feat['fav_cnt'] + feat['cart_cnt'] + 1)
    feat['cart_ratio'] = feat['cart_cnt'] / (feat['view_cnt'] + feat['fav_cnt'] + feat['cart_cnt'] + 1)
    # 最近一次交互距标签日的天数（用 feat_end 为参照）
    feat['last_gap_days'] = feat['last_time'].apply(lambda x: (pd.to_datetime(feat_end) - pd.to_datetime(x)).days if not pd.isna(x) else 999)

    # 与标签合并（左连接，未出现的 label 为 0）
    dataL = feat.merge(label_df[['user_id','item_id','label']], on=['user_id','item_id'], how='left')
    dataL['label'] = dataL['label'].fillna(0).astype(int)

    return dataL, label_df

# ========== 样本平衡函数（欠采样负样本） ==========
def balance_samples(df, neg_pos_ratio=NEG_POS_RATIO, random_state=RANDOM_STATE):
    """
    df: 包含 label 的 DataFrame
    返回： 欠采样后的 df
    """
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    if len(pos) == 0:
        # 没有正样本，直接返回原 df（后面模型训练会跳过）
        return df
    n_neg = min(len(neg), int(len(pos) * neg_pos_ratio))
    neg_sample = neg.sample(n=n_neg, random_state=random_state) if n_neg>0 else neg
    balanced = pd.concat([pos, neg_sample], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced

# ========== 窗口训练 & 交叉验证（滑动窗口） ==========
print("2) 开始滑动窗口训练（每次用前6天预测第7天，并计算F1/AUC）...")
window_start = min_date
window_end_for_start = max_date - timedelta(days=WINDOW_DAYS)  # 最后一个可用于预测的起始日

window_f1s = []
window_aucs = []
window_counts = []

# 取滑动区间
cur = window_start
while cur <= window_end_for_start:
    try:
        dataL, label_df = build_features_labels(cur, df, window_days=WINDOW_DAYS)
    except Exception as e:
        print("窗口构造异常，跳过：", cur, e)
        cur = cur + timedelta(days=1)
        continue

    if dataL is None or dataL.shape[0] == 0:
        cur = cur + timedelta(days=1)
        continue

    # 平衡样本
    balanced = balance_samples(dataL, neg_pos_ratio=NEG_POS_RATIO, random_state=RANDOM_STATE)

    # 准备训练集 X,y
    feat_cols = [
        'view_cnt','fav_cnt','cart_cnt','buy_cnt',
        'time_weighted_score','category_purchase_adjust','intent_adjust','global_pref_score',
        'buy_ratio','cart_ratio','last_gap_days'
    ]
    # 有些列在某些窗口可能不存在（比如 global_pref_score），用 fillna
    X = balanced[feat_cols].fillna(0)
    y = balanced['label'].values

    # 如果没有正样本或负样本就跳过
    if y.sum() == 0 or y.sum() == len(y):
        cur = cur + timedelta(days=1)
        continue

    # 训练一个短时 LightGBM（用默认LGB_PARAMS）
    clf = LGBMClassifier(**LGB_PARAMS)
    clf.fit(X, y)

    # 验证：用第7天的真实标签（label_df）构造特征并预测（注意：build_features_labels 函数已经把第7天的 label 与特征合并为 dataL）
    # dataL 中包含 label 和相应的 feature（基于窗口特征聚合）
    # 为验证，使用 balanced 中未参与训练的那部分? 简化：直接在整个 dataL 上评估（因为我们采用滑动窗口作为近似的时序验证）
    X_all = dataL[feat_cols].fillna(0)
    y_all = dataL['label'].values
    y_pred_proba = clf.predict_proba(X_all)[:,1]
    # 选择阈值 0.35（可以后面自动调整）
    y_pred = (y_pred_proba >= THRESHOLD_FINAL).astype(int)

    # 评估
    try:
        f1 = f1_score(y_all, y_pred)
    except Exception:
        f1 = 0.0
    try:
        auc = roc_auc_score(y_all, y_pred_proba)
    except Exception:
        auc = 0.0

    window_f1s.append(f1)
    window_aucs.append(auc)
    window_counts.append(len(dataL))

    print(f"窗口起始 {cur}  | 样本数 {len(dataL)} | pos={y_all.sum()} | F1={f1:.5f} | AUC={auc:.5f}")

    cur = cur + timedelta(days=1)

# 打印滑动窗口总体表现
if window_f1s:
    print("滑动窗口平均 F1:", np.mean(window_f1s), "平均 AUC:", np.mean(window_aucs))
else:
    print("未生成有效窗口用于训练/评估，请检查时间范围或数据。")

# ========== 最终训练：用最后 6 天训练并预测 12月19日 ==========
print("3) 使用最后6天训练模型，预测 2014-12-19 ...")
# 定位最后可用起始日，使得第7天为 max_date (应该是 2014-12-19)
final_window_start = max_date - timedelta(days=WINDOW_DAYS)  # 如果 max_date 是 12-19，则 start = 12-13
train_dataL, train_label_df = build_features_labels(final_window_start, df, window_days=WINDOW_DAYS)
if train_dataL is None:
    print("最终窗口构造失败，退出。")
    sys.exit(1)

balanced_final = balance_samples(train_dataL, neg_pos_ratio=NEG_POS_RATIO, random_state=RANDOM_STATE)

feat_cols = [
    'view_cnt','fav_cnt','cart_cnt','buy_cnt',
    'time_weighted_score','category_purchase_adjust','intent_adjust','global_pref_score',
    'buy_ratio','cart_ratio','last_gap_days'
]
X_final = balanced_final[feat_cols].fillna(0)
y_final = balanced_final['label'].values

# 训练最终模型
final_clf = LGBMClassifier(**LGB_PARAMS)
final_clf.fit(X_final, y_final)

# 构造预测候选集：这里我们采用最后6天的交互中出现过的 user-item 对作为候选
predict_window_start = final_window_start
predict_window_end = final_window_start + timedelta(days=WINDOW_DAYS-1)
candidates = df[(df['date'] >= predict_window_start) & (df['date'] <= predict_window_end)][['user_id','item_id']].drop_duplicates()

# 构造候选的特征（基于同一函数思想，但我们需要快速构造）
# 为了复用，我们构造一个临时 df 包含 candidates 的特征聚合
temp_df = df[(df['date'] >= predict_window_start) & (df['date'] <= predict_window_end)].copy()
# 聚合同 build_features_labels 的特征步骤（简化）
temp_df['is_view'] = (temp_df['behavior_type']==1).astype(int)
temp_df['is_fav']  = (temp_df['behavior_type']==2).astype(int)
temp_df['is_cart'] = (temp_df['behavior_type']==3).astype(int)
temp_df['is_buy']  = (temp_df['behavior_type']==4).astype(int)
base = temp_df.groupby(['user_id','item_id'], as_index=False).agg(
    view_cnt = ('is_view','sum'),
    fav_cnt  = ('is_fav','sum'),
    cart_cnt = ('is_cart','sum'),
    buy_cnt  = ('is_buy','sum'),
    last_time = ('time','max')
)
# 时间加权
ev = temp_df[temp_df['behavior_type'].isin([1,2,3,4])].copy()
ev['days_diff'] = ((pd.to_datetime(predict_window_end) - ev['time']).dt.total_seconds() / (3600*24)).clip(lower=0)
decay_days = 3.0
ev['time_weight'] = np.exp(-ev['days_diff'] / decay_days)
ev['type_weight'] = ev['behavior_type'].map(BEHAVIOR_WEIGHT).fillna(1.0)
ev['weighted'] = ev['time_weight'] * ev['type_weight']
uw = ev.groupby(['user_id','item_id'], as_index=False)['weighted'].sum().rename(columns={'weighted':'time_weighted_score'})
feat_cand = base.merge(uw, on=['user_id','item_id'], how='left')
feat_cand['time_weighted_score'] = feat_cand['time_weighted_score'].fillna(0.0)
# category adj based on window (simpler: set 0)
feat_cand['category_purchase_adjust'] = 0
# intent adjust (approx using window)
user_fav = temp_df[temp_df['behavior_type'] == 2].groupby('user_id')['item_id'].nunique().rename('fav_total')
user_fav_buy = temp_df[temp_df['behavior_type'] == 4].groupby('user_id')['item_id'].nunique().rename('buy_total')
user_fav_merge = pd.concat([user_fav, user_fav_buy], axis=1).fillna(0)
user_fav_merge['fav_buy_ratio'] = np.where(user_fav_merge['fav_total'] > 0,
                                           user_fav_merge['buy_total'] / user_fav_merge['fav_total'], 0)
user_cart = temp_df[temp_df['behavior_type'] == 3].groupby('user_id')['item_id'].nunique().rename('cart_total')
user_cart_buy = temp_df[temp_df['behavior_type'] == 4].groupby('user_id')['item_id'].nunique().rename('buy_total2')
user_cart_merge = pd.concat([user_cart, user_cart_buy], axis=1).fillna(0)
user_cart_merge['cart_buy_ratio'] = np.where(user_cart_merge['cart_total'] > 0,
                                             user_cart_merge['buy_total2'] / user_cart_merge['cart_total'], 0)
intent_tmp = pd.concat([user_fav_merge['fav_buy_ratio'], user_cart_merge['cart_buy_ratio']], axis=1).fillna(0)
intent_tmp['intent_score'] = 0.4 * intent_tmp['fav_buy_ratio'] + 0.6 * intent_tmp['cart_buy_ratio']
def intent_adj_tmp(x):
    if x > 0.7: return 0.2
    if x < 0.3: return -0.2
    return 0.0
intent_tmp['intent_adjust'] = intent_tmp['intent_score'].apply(intent_adj_tmp)
feat_cand = feat_cand.merge(intent_tmp['intent_adjust'], left_on='user_id', right_index=True, how='left')
feat_cand['intent_adjust'] = feat_cand['intent_adjust'].fillna(0.0)
# global pref (window)

if 'item_category' not in feat_cand.columns:
    feat_cand = feat_cand.merge(
        items[['item_id', 'item_category']].drop_duplicates(),
        on='item_id',
        how='left'
    )
global_pref = (temp_df[temp_df['behavior_type']==4].groupby('item_category')['item_id'].count().rename('global_buy_count'))
if not global_pref.empty:
    gp = (global_pref / global_pref.max()).rename('global_pref_score').reset_index()
    feat_cand = feat_cand.merge(gp, on='item_category', how='left')
    feat_cand['global_pref_score'] = feat_cand['global_pref_score'].fillna(0.0)
else:
    feat_cand['global_pref_score'] = 0.0

# 衍生
feat_cand['buy_ratio'] = feat_cand['buy_cnt'] / (feat_cand['view_cnt'] + feat_cand['fav_cnt'] + feat_cand['cart_cnt'] + 1)
feat_cand['cart_ratio'] = feat_cand['cart_cnt'] / (feat_cand['view_cnt'] + feat_cand['fav_cnt'] + feat_cand['cart_cnt'] + 1)
feat_cand['last_gap_days'] = feat_cand['last_time'].apply(lambda x: (pd.to_datetime(predict_window_end) - pd.to_datetime(x)).days if not pd.isna(x) else 999)

# 预测
X_cand = feat_cand[feat_cols].fillna(0)
feat_cand['score'] = final_clf.predict_proba(X_cand)[:,1]

# 按 user_id 排序，取每个用户 top5，且过滤阈值
feat_cand_sorted = feat_cand.sort_values(['user_id','score'], ascending=[True,False])
feat_top = feat_cand_sorted.groupby('user_id').head(5).reset_index(drop=True)

# 根据阈值筛选高置信度
final_selected = feat_top[feat_top['score'] >= THRESHOLD_FINAL].copy()

# 输出提交文件（只包含 user_id, item_id）
ensure_path_dir(OUTPUT_SUBMIT)
final_selected[['user_id','item_id']].to_csv(OUTPUT_SUBMIT, index=False)
print(f"提交文件已保存到: {OUTPUT_SUBMIT} (共 {len(final_selected)} 条记录)")

# 备份含 score 的文件，便于分析
ensure_path_dir(OUTPUT_WITH_SCORE)
final_selected[['user_id','item_id','score']].to_csv(OUTPUT_WITH_SCORE, index=False)
print(f"含 score 的文件已保存到: {OUTPUT_WITH_SCORE}")

print("全部完成。")
