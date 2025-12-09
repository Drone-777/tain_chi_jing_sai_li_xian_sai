import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn import svm

# ==============================================================================
# 0. 常量定义
# ==============================================================================
USER_FILE = 'tianchi_fresh_comp_train_user.csv'
ITEM_FILE = 'tianchi_fresh_comp_train_item.csv'
PREDICT_FILE = 'tianchi_mobile_recommendation_predict.csv'

def run_tianchi_pipeline():
    """
    整合用户提供的所有数据处理、特征工程和模型训练步骤。
    """
    print("🚀 开始执行天池新人赛数据分析及模型训练流水线...")

    # ==============================================================================
    # 1. 数据筛选 (对应第一部分代码)
    # ==============================================================================
    print("\n[Step 1/6] 载入数据并进行初步筛选...")
    try:
        df = pd.read_csv(USER_FILE)
        itemP = pd.read_csv(ITEM_FILE)
    except FileNotFoundError as e:
        print(f"错误：未能找到文件。请确保 {USER_FILE} 和 {ITEM_FILE} 存在于当前目录。")
        raise e

    # 清理 geohash 列
    if "user_geohash" in df.columns:
        del df["user_geohash"]
    if "item_geohash" in itemP.columns:
        del itemP['item_geohash']

    # 处理时间列：格式化并删除 2014-12-12 的数据
    df['time'] = df['time'].astype(str).str.slice(0, 10)
    # 保留所有非 '2014-12-12' 的数据
    df = df.loc[df['time'] != '2014-12-12'].copy()

    # 商品 ID 筛选：只保留出现在 itemP 中的商品
    itemsub = set(itemP['item_id'].astype(str))
    df['item_id'] = df['item_id'].astype(str)
    
    # 使用 isin 进行高效筛选
    df = df.loc[df['item_id'].isin(itemsub)].copy()

    print(f"   数据筛选完成，剩余 {len(df)} 条记录。")


    # ==============================================================================
    # 2. 哑编码 (对应第二部分代码)
    # ==============================================================================
    print("\n[Step 2/6] 进行行为类型 One-Hot 编码...")
    
    # 对 'behavior_type' 列进行 One-Hot 编码
    # behavior_type 1=look, 2=like, 3=putin, 4=buy
    one_hot = pd.get_dummies(df['behavior_type'])
    one_hot.rename(columns={1: 'look', 2: 'like', 3: 'putin', 4: 'buy'}, inplace=True)
    
    # 合并 One-Hot 编码结果并清理
    df = pd.concat([df.reset_index(drop=True), one_hot.reset_index(drop=True)], axis=1)
    
    # 删除原始行为类型列
    del df['behavior_type']
    # 删除在第 1 部分中用于临时标记的 'item_mark' 列（实际上原代码未创建，但为保证逻辑兼容性保留删除操作）
    if 'item_mark' in df.columns:
        del df['item_mark'] 
    
    # 确保 time 列是 datetime 类型，以便后续处理
    df['time'] = pd.to_datetime(df['time'])

    print("   One-Hot 编码完成，创建了 look, like, putin, buy 四个行为列。")

    # ==============================================================================
    # 3. 标记日期 (对应第三部分代码)
    # ==============================================================================
    print("\n[Step 3/6] 创建日期特征 time_mark 和 2days...")
    
    # 优化后的日期标记：标记 12月17日 和 12月18日
    df['time_mark'] = np.where(
        (df['time'].dt.month == 12) & (df['time'].dt.day.isin([17, 18])),
        1,
        0
    )

    # 标记 19 号前两天发生的购买数据 (12月17日或18日的购买行为)
    df['2days'] = df['buy'] * df['time_mark']
    
    print("   日期标记完成。")

    # ==============================================================================
    # 4. 加权 (对应第四部分代码)
    # ==============================================================================
    print("\n[Step 4/6] 计算加权特征 wight...")
    
    # 使用原代码中的统计量进行加权计算（这些统计量应是基于整个训练集的）
    # look.sum() = 1863827, like.sum() = 32506, putin.sum() = 53646, buy.sum() = 20989
    buy_count = 20989
    
    # 计算基础权重
    W_look = buy_count / 1863827
    W_like = buy_count / 32506
    W_putin = buy_count / 53646
    
    # 计算加权特征
    df['wight'] = (
        df['look'] * W_look + 
        df['like'] * W_like + 
        df['putin'] * W_putin + 
        df['buy'] +             # W_buy = 1
        df['time_mark']         # 时间标记直接加权
    ) * ((2 - df['2days']) / 2) # 12-17/18 的购买行为惩罚 (权重减半)

    print("   特征加权 wight 计算完成。")


    # ==============================================================================
    # 5. 样本筛选 (对应第五部分代码)
    # ==============================================================================
    print("\n[Step 5/6] 筛选样本：排除浏览操作 (look != 1)...")
    
    # 筛选出所有非浏览行为的样本，作为模型的训练数据
    df_model = df.loc[df['look'] != 1].copy()
    
    # 清理列：删除 'look' 列
    del df_model['look']
    
    print(f"   样本筛选完成，用于模型训练的样本数为 {len(df_model)}。")


    # ==============================================================================
    # 6. 模型训练与预测 (对应第六部分代码)
    # ==============================================================================
    print("\n[Step 6/6] SVM 模型训练、评估与预测...")
    
    # 设置 'time' 列为索引 (与原代码保持一致)
    df_model.set_index('time', inplace=True)
    
    df_model['label_y'] = df_model['buy']	# 以购买操作作为标记 (目标变量)

    # --- 准备训练数据 ---
    # 修正：使用 .loc 替换已弃用的 .ix
    feature_cols = ['user_id', 'item_id', 'item_category', 'putin', 'buy', 'time_mark', 'wight']
    X = df_model.loc[:, feature_cols]
    y = df_model['label_y']
    
    # 分割数据集用于模型评估
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 初始化 SVM 模型
    # 注意：对于大型数据集，SVM (SVC) 训练速度非常慢。建议在实际比赛中使用更高效的分类器 (如 LightGBM 或 XGBoost)。
    clf = svm.SVC(C=100, class_weight='balanced', random_state=42)
    
    # 训练模型
    print("   正在训练 SVM 模型 (这可能需要较长时间)...")
    clf.fit(X_train, y_train)
    
    # 预测并评估
    predict = clf.predict(X_test)
    print("\n--- 模型评估结果 ---")
    print(f"   Accuracy Score: {clf.score(X_test, y_test):.4f}")
    print("   Classification Report:")
    print(classification_report(y_test, predict))
    print(f"   Weighted F1 Score: {f1_score(y_test, predict, average='weighted'):.4f}")
    print("----------------------")


    # --- 准备预测目标数据 ---
    # 目标：预测 2014-12-18 当天有 'putin' (加入购物车) 行为的用户是否会在 12-19 购买
    # 修正：使用 .loc 替换 .ix
    outputSet = df_model.loc['2014-12-18'].copy()
    # 筛选出 12-18 当天有加入购物车 (putin=1) 的记录
    outputSet = outputSet.loc[outputSet['putin'] == 1].copy()

    # 准备预测特征 (与训练特征保持一致)
    X_predict = outputSet.loc[:, feature_cols]
    
    # 预测
    print("   正在对目标数据进行预测...")
    output = clf.predict(X_predict)
    X_predict['output'] = output
    
    # 筛选出预测结果为“购买” (output > 0.0) 的记录
    X_predict = X_predict.loc[X_predict['output'] > 0.0].copy()

    # 提取最终结果： user_id, item_id
    final_result = X_predict.loc[:, ['user_id', 'item_id']].copy()

    # 删除 time 索引，将其变为普通列
    if 'time' in final_result.index.names:
        final_result.reset_index(inplace=True)
        if 'time' in final_result.columns:
            del final_result['time']

    # 保存结果
    final_result.to_csv(PREDICT_FILE, index=False)
    
    print(f"\n🎉 预测完成！结果已保存到文件：{PREDICT_FILE}")

if __name__ == '__main__':
    run_tianchi_pipeline()