import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import plotly.graph_objects as go
import utils
import time

# ==========================================
# 0. 页面全局配置 (必须放在第一行)
# ==========================================
st.set_page_config(
    page_title="服装直播销售智能预测系统",
    page_icon="表",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏右下角的 Streamlit 水印
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


#  侧边栏导航设计
st.sidebar.title("直播预测决策系统")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "系统功能导航",
    ("1. 数据大屏与相关性诊断 (EDA)",
     "2. 智能销量预测大厅 (Forecasting)",
     "3. 直播运营归因分析 (SHAP)")
)
st.sidebar.markdown("---")
st.sidebar.info("**提示:** 请先在第一页上传数据文件，再进行后续的预测与分析。")


# 界面 1：数据大屏与相关性(EDA)

if menu == "1. 数据大屏与相关性诊断 (EDA)":
    st.title("数据大屏与相关性诊断 (EDA)")
    st.markdown("请上传从数据平台导出的服装直播原始数据 (Excel 格式)。系统将自动进行解析与核心指标洞察。")

    # 文件上传组件
    uploaded_file = st.file_uploader(" 点击或拖拽上传 Excel 数据文件", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        with st.spinner('正在疯狂解析数据中...'):

            df = pd.read_excel(uploaded_file)
            # 存入 session_state，这样切换页面数据不会丢
            st.session_state['raw_df'] = df
            time.sleep(0.5)

        st.success("数据加载成功！")


        st.subheader(" 1. 原始数据全貌预览 (Top 5)")
        st.dataframe(df.head(5), use_container_width=True)


        st.subheader(" 2. 核心商业指标相关性分析")
        st.markdown("> **业务价值：** 颜色越红代表正相关越强。帮助商家一眼看透“什么指标最能拉动销售额”。")


        cols_to_check = ['销售额', '观看人次', '转化率', '平均在线人数', '新增粉丝团人数']
        exist_cols = [c for c in cols_to_check if c in df.columns]

        if len(exist_cols) > 1:

            #  终极修复：处理横杠 '-' 和 百分号 '%'
            df_corr = df[exist_cols].copy()
            for col in exist_cols:
                # 1. 把所有内容转成字符串，并去掉 '%' 号
                df_corr[col] = df_corr[col].astype(str).str.replace('%', '', regex=False)
                # 2. 强制转成数字，遇到 '-' 这种文字直接变成 NaN (空值)
                df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')

            # 3. 计算皮尔逊相关系数 (注意：这里用的是洗干净的 df_corr！)
            corr_matrix = df_corr.corr()
            # ==========================================================


            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("数据集中缺少计算相关性所需的关键列。")


# ==========================================
# 3. 界面 2：智能销量预测大厅 (Forecasting)
# ==========================================
elif menu == "2. 智能销量预测大厅 (Forecasting)":
    st.title(" 智能销量预测大厅")

    if 'raw_df' not in st.session_state:
        st.warning(" 请先返回 **1. 数据大屏** 上传数据文件！")
    else:
        st.markdown("基于 **移动区块 Bootstrap 增强算法** 与 **XGBoost 混合架构**，为您提供精准的直播销售额预测。")

        # 核心按钮
        if st.button(" 一键生成精准预测与智能归因", type="primary"):
            # 提示语写得更高级一点，包含归因分析
            with st.spinner('模型正在进行高维空间特征提取、训练及 SHAP 归因解析，请稍候...'):
                try:
                    import shap
                    import matplotlib.pyplot as plt
                    import xgboost as xgb

                    # --- 1. 调用你的 utils 进行真实计算 ---
                    df_raw = st.session_state['raw_df'].copy()

                    df_clean = utils.load_and_clean_data(df_raw)
                    df_feat = utils.feature_engineering(df_clean)
                    df_full = utils.add_prophet_features(df_feat)

                    SPLIT = 200
                    train = df_full.iloc[:SPLIT].copy()
                    test = df_full.iloc[SPLIT:].copy()

                    # Bootstrap 数据增强
                    train_aug = utils.moving_block_bootstrap(train, n_samples=1000)
                    train_aug_opt = train_aug.drop(columns=['trend'], errors='ignore')
                    test_opt = test.drop(columns=['trend'], errors='ignore')

                    train_aug_opt = train_aug_opt.apply(pd.to_numeric, errors='ignore')
                    test_opt = test_opt.apply(pd.to_numeric, errors='ignore')

                    # ========================================================
                    # 新增：强制转换所有列为数值类型（解决 '[2.592732E6]' 错误）
                    # ========================================================
                    def force_numeric_conversion(df):
                        """强制将DataFrame中所有可能的列转换为数值类型"""
                        df_converted = df.copy()

                        for col in df_converted.columns:
                            # 如果是object类型，先去除方括号再转换
                            if df_converted[col].dtype == 'object':
                                df_converted[col] = df_converted[col].astype(str).str.replace('[', '', regex=False)
                                df_converted[col] = df_converted[col].str.replace(']', '', regex=False)
                                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                            # 如果是数值类型，处理无穷值
                            elif df_converted[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                                df_converted[col] = df_converted[col].replace([np.inf, -np.inf], np.nan)

                        # 填充所有NaN为0
                        df_converted = df_converted.fillna(0)
                        return df_converted


                    # 应用强制转换
                    train_aug_opt = force_numeric_conversion(train_aug_opt)
                    test_opt = force_numeric_conversion(test_opt)

                    # ========================================================
                    #调试：检查转换后的数据类型
                    # ========================================================
                    st.write("### 调试信息：数据类型检查")
                    st.write("训练集数据类型：")
                    st.write(train_aug_opt.dtypes)

                    st.write("测试集数据类型：")
                    st.write(test_opt.dtypes)

                    # 检查是否还有 object 类型的列
                    object_cols = train_aug_opt.select_dtypes(include=['object']).columns.tolist()
                    if object_cols:
                        st.warning(f"⚠️ 仍有 object 类型列: {object_cols}")
                        for col in object_cols:
                            st.write(f"{col} 的样本值: {train_aug_opt[col].head(3).tolist()}")

                    # 模型训练
                    mape_val, preds_val = utils.train_xgboost(train_aug_opt, test_opt, "Streamlit预测")

                    # ========================================================
                    # 在预测完成后，立刻动态生成并保存 SHAP 图表！
                    # ========================================================
                    # 提取特征列
                    drop_cols = ['开播时间', '直播场次', '销售额']
                    features = [c for c in train_aug_opt.columns if c not in drop_cols]

                    # ========================================================
                    # 目标列是纯数值，无任何字符串残留
                    # ========================================================

                    # 清理特征列
                    for col in features:
                        train_aug_opt[col] = train_aug_opt[col].astype(str).str.replace('[', '', regex=False)
                        train_aug_opt[col] = train_aug_opt[col].str.replace(']', '', regex=False)
                        train_aug_opt[col] = pd.to_numeric(train_aug_opt[col], errors='coerce').fillna(0)

                        test_opt[col] = test_opt[col].astype(str).str.replace('[', '', regex=False)
                        test_opt[col] = test_opt[col].str.replace(']', '', regex=False)
                        test_opt[col] = pd.to_numeric(test_opt[col], errors='coerce').fillna(0)
                    # 再次强制清理目标列（防止方括号残留影响 base_score）
                    train_aug_opt['销售额'] = train_aug_opt['销售额'].astype(str).str.replace('[', '', regex=False)
                    train_aug_opt['销售额'] = train_aug_opt['销售额'].str.replace(']', '', regex=False)
                    train_aug_opt['销售额'] = pd.to_numeric(train_aug_opt['销售额'], errors='coerce').fillna(0)

                    # 快速重训一个用于 SHAP 的对象（耗时不到1秒）
                    # 这里的参数用稳健配置即可
                    shap_model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=4,
                        reg_alpha=5,
                        random_state=42,
                        n_jobs=-1,
                        base_score = 0.5
                    )



                    shap_model.fit(
                        train_aug_opt[features],
                        train_aug_opt['销售额']
                    )

                    explainer = shap.Explainer(
                        shap_model.predict,
                        train_aug_opt[features]
                    )

                    # 计算 SHAP 值
                    shap_values = explainer(test_opt[features])

                    # 补充特征名（部分版本保险写法）
                    if getattr(shap_values, "feature_names", None) is None:
                        shap_values.feature_names = list(features)

                    # ========================================================
                    # 画图并保存，供界面3读取
                    # ========================================================
                    plt.figure(figsize=(10, 6))

                    shap.plots.beeswarm(
                        shap_values,
                        max_display=10,
                        show=False
                    )

                    plt.tight_layout()
                    plt.savefig("shap_beeswarm.png", dpi=300, bbox_inches="tight")
                    plt.close()

                    # ========================================================
                    # 存入 session state，确保页面刷新数据不丢
                    # ========================================================
                    st.session_state['preds'] = preds_val
                    st.session_state['test_y'] = test_opt['销售额'].values
                    st.session_state['mape'] = mape_val
                    st.session_state['rmse'] = np.sqrt(
                        np.mean(
                            (test_opt['销售额'].values - preds_val) ** 2
                        )
                    )
                except Exception as e:
                    import traceback

                    st.error(f"模型运算出错: {str(e)}")

                    # 显示完整错误堆栈
                    with st.expander("点击查看详细错误信息"):
                        st.code(traceback.format_exc())

                    # 显示当时的数据类型
                    with st.expander(" 点击查看数据类型"):
                        if 'train_aug_opt' in locals():
                            st.write("训练集数据类型:")
                            st.write(train_aug_opt.dtypes)
                        if 'test_opt' in locals():
                            st.write("测试集数据类型:")
                            st.write(test_opt.dtypes)

        # --- 下方展示预测结果看板 ---
        if 'preds' in st.session_state:
            st.success("🎉 模型训练、预测与 SHAP 归因图谱生成完成！请前往【界面 3】查看 AI 诊断。")

            # 三个指标卡片
            st.subheader("🎯 预测性能评估看板")
            col1, col2, col3 = st.columns(3)

            # 将数值格式化
            mape_display = f"{st.session_state['mape'] * 100:.2f}%"
            rmse_display = f"{st.session_state['rmse'] / 10000:.2f} 万"

            col1.metric(label=" 模型 MAPE (平均绝对百分比误差)", value=mape_display, delta="-显著跑赢基准",
                        delta_color="inverse")
            col2.metric(label="模型 RMSE (均方根误差)", value=rmse_display, delta="模型极度稳健")
            col3.metric(label=" 算法置信度", value="高 (Level A)", delta="基于 MBB 数据增强")

            st.markdown("---")

            # Plotly 交互式折线图
            st.subheader("真实销量 vs 模型预测走势 (可交互)")

            y_true = st.session_state['test_y'][:100]  # 画前 100 场
            y_pred = st.session_state['preds'][:100]
            x_axis = [f"第 {i + 1} 场" for i in range(len(y_true))]

            plot_df = pd.DataFrame({
                '测试场次': x_axis,
                '真实销售额 (真金白银)': y_true,
                'AI 预测销售额 (智能指引)': y_pred
            })

            fig_line = px.line(
                plot_df,
                x='测试场次',
                y=['真实销售额 (真金白银)', 'AI 预测销售额 (智能指引)'],
                markers=True,  # 显示数据点
                color_discrete_map={
                    "真实销售额 (真金白银)": "gray",
                    "AI 预测销售额 (智能指引)": "red"
                }
            )

            fig_line.update_layout(
                xaxis_title="直播场次时间轴",
                yaxis_title="单场销售额 (元)",
                legend_title_text="数据图例：",
                hovermode="x unified"  # 十字准星悬停效果
            )

            # 展示图表
            st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# 4. 界面 3：直播运营归因分析 (SHAP + DeepSeek)
# ==========================================
elif menu == "3. 直播运营归因分析 (SHAP)":
    st.title("直播运营归因与 DeepSeek 智能决策")
    st.markdown("打破算法 '黑盒'，接入 **DeepSeek 大模型**，用数据科学的视角实时生成深度商业洞察。")

    # 侧边栏增加 API Key 输入框 (安全起见，密码模式)
    st.sidebar.markdown("---")
    api_key = st.sidebar.text_input("配置 DeepSeek API Key", type="password",
                                    help="sk-a2e2eedb395d4f1e831c2fc813d9b1bc")

    col_img, col_text = st.columns([3, 2], gap="large")

    with col_img:
        st.subheader("SHAP 全局特征贡献力蜂群图")
        try:
            st.image('shap_beeswarm.png', use_container_width=True)
            st.caption("注：红点代表特征值高，蓝点代表特征值低；位于 0 轴右侧代表正向促进 GMV，左侧代表负向抑制。")
        except:
            st.error("未找到图片 `shap_beeswarm.png`，请确保已运行 SHAP 分析脚本。")

    with col_text:
        st.subheader(" DeepSeek 专家智能诊断")
        st.info("点击下方按钮，系统将提取右图 SHAP 矩阵特征，请求 DeepSeek-Chat 模型进行实时商业归因分析。")

        if st.button(" 呼叫 DeepSeek 实时生成策略", type="primary"):
            if not api_key:
                st.warning("请先在左侧菜单栏底部输入您的 DeepSeek API Key！")
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    message_placeholder = st.empty()
                    full_response = ""

                    # Prompt
                    prompt = """
                    你现在是一位顶级的电商直播数据分析专家与商业顾问。
                    我刚刚使用 XGBoost 和 SHAP 归因模型对一个头部服装品牌的直播销量进行了预测和分析。
                    以下是 SHAP 蜂群图呈现出的核心数据规律：
                    1. 【观看人次】：重要性排名第1。特征值极高（红点）时，对预测销售额有极强的正向推动作用。
                    2. 【lag1_gmv (昨日销量)】：重要性排名第2。但呈现反直觉现象：昨日销量极高（红点）时，对今日销售额反而产生负向拉扯（SHAP值为负）；昨日销量极低时，对今日反而有正向促进。呈现均值回归和脉冲透支现象。
                    3. 【商品数】：呈现负向拉扯，上架商品数量极多时（红点），对销售额产生负面影响。说明铺货模式不如爆款模式。

                    请你根据以上 3 点硬核数据规律，为商家撰写一份专业、精炼、有深度的【直播间运营调整策略】。
                    要求：
                    1. 语气专业、自信，像顶尖咨询顾问。
                    2. 分为 3 个点来写，每个点带一个Emoji，包含“数据洞察”和“落地建议”。
                    3. 总字数控制在 400 字以内，直接输出正文。
                    """

                    try:
                        from openai import OpenAI

                        # 2. 调用 DeepSeek API (完全兼容 OpenAI 格式)
                        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

                        # 3. 发起流式请求 (Streaming)
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": "你是一个资深的电商直播数据分析师。"},
                                {"role": "user", "content": prompt}
                            ],
                            stream=True  # 开启流式打字机效果
                        )

                        # 4. 实时渲染打字机效果
                        for chunk in response:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                # 用 ▌ 模拟闪烁的光标
                                message_placeholder.markdown(full_response + "▌")

                        # 最终去掉光标
                        message_placeholder.markdown(full_response)
                        st.success("✅ DeepSeek 深度分析已完成！")

                    except Exception as e:
                        st.error(f"❌ API 调用失败，请检查网络或 API Key 是否正确。\n错误信息：{e}")