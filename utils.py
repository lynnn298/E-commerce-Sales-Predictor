import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据清洗
def load_and_clean_data(file_input):
    print(">>>[1/5] 正在加载并清洗数据...")

    if isinstance(file_input, str):
        # 如果传进来的是一段字符串路径，就去硬盘读
        df = pd.read_excel(file_input)
    else:
        # 如果传进来的已经是 DataFrame（比如Streamlit传来的），就直接复制一份用
        df = file_input.copy()
    # =========================================

    # 强制转换时间格式
    df['开播时间'] = pd.to_datetime(df['开播时间'], errors='coerce')
    df = df.dropna(subset=['开播时间'])
    df = df.sort_values('开播时间').reset_index(drop=True)

    # 强制将数值列转换为数字
    numeric_cols = ['销售额', '销量', '观看人次', '商品数', '直播时长']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)

    threshold = 500
    df_clean = df[df['销售额'] > threshold].copy()

    keep_cols = ['开播时间', '直播场次', '销售额', '观看人次', '商品数']
    df_clean = df_clean[[c for c in keep_cols if c in df_clean.columns]]

    print(f" 清洗完毕，剩余 {len(df_clean)} 条有效数据")
    return df_clean.reset_index(drop=True)



#df = load_and_clean_data(r"D:\毕设\蝉妈妈数据\按店铺来的数据\罗蒙官方旗舰店\罗蒙官方旗舰店_2022-02-18-2025-02-21_达人详情直播记录.xlsx")


# 特征工程
def feature_engineering(df):
    data = df.copy()

    data['date_only'] = data['开播时间'].dt.date

    data['month'] = data['开播时间'].dt.month
    data['day_of_week'] = data['开播时间'].dt.dayofweek
    data['hour'] = data['开播时间'].dt.hour


    def is_promo(row):
        date = row['开播时间']
        title = str(row['直播场次'])

        if (date.month == 11 and date.day == 11) or (date.month == 6 and date.day == 18) or (date.month == 1):
            return 1
        if any(x in title for x in ['大促', '狂欢', '年货', '双11', '双12']):
            return 1
        return 0

    data['is_promo'] = data.apply(is_promo, axis=1)

    def check_season(title, keywords):
        for k in keywords:
            if k in str(title): return 1
        return 0

    #季节
    data['is_winter'] = data['直播场次'].apply(lambda x: check_season(x, ['羽绒', '加厚', '冲锋衣', '棉服', '毛呢', '冬']))
    data['is_summer'] = data['直播场次'].apply(
        lambda x: check_season(x, ['短袖', '冰丝', 'Polo', 'T恤', '防晒', '薄款', '夏']))

    data['is_star'] = data['直播场次'].apply(lambda x: check_season(x, ['林志颖', '明星', '同款']))
    data['is_spring_autumn'] = data['直播场次'].apply(
        lambda x: check_season(x, ['夹克', '早春', '秋季']))

    # 历史滞后特征

    # 上一场
    df = df.sort_values('开播时间')
    data['last_session_gmv'] = data['销售额'].shift(1)

    daily_stats = data.groupby('date_only')['销售额'].sum().reset_index()
    daily_stats.columns = ['date_only', 'daily_total_gmv']

    # 昨天上周
    daily_stats['lag1_daily_gmv'] = daily_stats['daily_total_gmv'].shift(1)
    daily_stats['lag7_daily_gmv'] = daily_stats['daily_total_gmv'].shift(7)

    # 计算最近3天的平均日销
    daily_stats['roll3_daily_mean'] = daily_stats['daily_total_gmv'].shift(1).rolling(window=3).mean()


    data = pd.merge(data, daily_stats[['date_only', 'lag1_daily_gmv', 'lag7_daily_gmv', 'roll3_daily_mean']],
                    on='date_only', how='left')

    # 过去3场的平均表现
    data['roll3_mean_gmv'] = data['销售额'].shift(1).rolling(window=3).mean()
    data['roll3_std_gmv'] = data['销售额'].shift(1).rolling(window=3).std()

    # 过去3场的流量表现
    if '观看人次' in data.columns:
        data['roll3_mean_uv'] = data['观看人次'].shift(1).rolling(window=3).mean()

    if '商品数' in data.columns:
        data['item_count'] = data['商品数']

    # data = data.dropna().reset_index(drop=True)
    # 昨天填充
    if 'lag7_daily_gmv' in data.columns:
        data['lag7_daily_gmv'] = data['lag7_daily_gmv'].fillna(data['lag1_daily_gmv'])

    # 填 0 (冷启动)
    fill_zero_cols = ['last_session_gmv', 'lag1_daily_gmv', 'lag7_daily_gmv',
                      'roll3_daily_mean', 'roll3_mean_gmv', 'roll3_mean_uv']

    for c in fill_zero_cols:
        if c in data.columns:
            data[c] = data[c].fillna(0)


    if 'date_only' in data.columns:
        data = data.drop(columns=['date_only'])


    return data





def moving_block_bootstrap(data, block_length=None, n_samples=1000):
    n = len(data)

    if block_length is None:
        block_length = max(2, int(n ** (1 / 3)))

    if n < block_length:
        return data.copy()

    new_data = []
    num_blocks = n - block_length + 1

    while len(new_data) * block_length < n_samples:
        start_idx = np.random.randint(0, num_blocks)
        block = data.iloc[start_idx:start_idx + block_length].copy()
        new_data.append(block)

    augmented_data = pd.concat(new_data, axis=0).reset_index(drop=True)

    # ======================================================
    # 🔥 关键修复：强制恢复数值类型
    # ======================================================
    for col in augmented_data.columns:
        augmented_data[col] = pd.to_numeric(
            augmented_data[col],
            errors='ignore'   # 不要直接 coerce（保留时间列等）
        )

    return augmented_data.iloc[:n_samples]



# 4. Prophet 特征提取

def add_prophet_features(df):
    print(">>> [Feature] 正在提取趋势与季节项特征 (统计学方法)...")
    data = df.copy()

    # 趋势
    data['trend'] = data['销售额'].shift(1).rolling(window=10, min_periods=1).mean()
    data['trend'] = data['trend'].fillna(0)

    # 周
    weekly_map = data.groupby('day_of_week')['销售额'].mean().to_dict()
    data['weekly'] = data['day_of_week'].map(weekly_map)

    # 月/年
    monthly_map = data.groupby('month')['销售额'].mean().to_dict()
    data['yearly'] = data['month'].map(monthly_map)

    # 日
    hourly_map = data.groupby('hour')['销售额'].mean().to_dict()
    data['daily'] = data['hour'].map(hourly_map)

    print("特征提取完成：已生成 trend, weekly, yearly, daily 列")
    return data



# 5. XGBoost 训练与评估函数

def train_xgboost(train_data, test_data, model_name="Model"):
    # ====== 在函数开头添加这段 ======
    # 确保输入数据都是数值类型
    train_data = train_data.copy()
    test_data = test_data.copy()

    for col in train_data.columns:
        if train_data[col].dtype == 'object':
            train_data[col] = pd.to_numeric(train_data[col].astype(str).str.replace('[', '').str.replace(']', ''),
                                            errors='coerce')
            test_data[col] = pd.to_numeric(test_data[col].astype(str).str.replace('[', '').str.replace(']', ''),
                                           errors='coerce')

    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    # ================================

    drop_cols = ['开播时间', '直播场次', '销售额']

    # 筛选特征列
    features = [c for c in train_data.columns if c not in drop_cols]
    target = '销售额'

    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

    # model = xgb.XGBRegressor(
    #     objective='reg:squarederror',
    #     n_estimators=120,
    #     learning_rate=0.08,
    #     max_depth=5,  #
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     reg_alpha=5, # 适度去噪
    #     reg_lambda=1,
    #
    #     min_child_weight=3,  # 一个叶子节点至少要有3个样本，防止被个别离群点带偏
    #     random_state=42,
    #     n_jobs=-1
    # )


    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,  # 原来6
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=100,
        reg_lambda=1,

        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)


    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds)

    print(f"--- [{model_name}] 结果 ---")
    print(f"使用特征数: {len(features)}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2%}")
    print("-" * 30)

    return mape, preds

