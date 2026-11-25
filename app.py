import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import root_mean_squared_error, roc_auc_score

st.set_page_config(page_title="Data Analysis & Competition App", layout="wide")

# --- メモリ軽量化関数（修正版） ---
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        # すべて欠損値の列・bool型はスキップ
        if df[col].isnull().all() or pd.api.types.is_bool_dtype(df[col]):
            continue

        # 整数型 (int) のみダウンキャスト
        if pd.api.types.is_integer_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        # 浮動小数点型 (float) のみダウンキャスト
        elif pd.api.types.is_float_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

        # 文字列型 (object / string) はユニーク数が少なければcategory化
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    return df, start_mem, end_mem

@st.cache_data
def load_file(uploaded_file):
    if uploaded_file.name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    return reduce_mem_usage(df)

# --- サイドバー：モード切替 ---
st.sidebar.title("⚙️ アプリ設定")
mode = st.sidebar.radio("モードを選択", ["通常分析モード", "コンペ用モード"])
st.title(f"📊 データ分析・MLアプリ（{mode}）")

# セッション状態の初期化
if "train_df" not in st.session_state:
    st.session_state.train_df = None
if "test_df" not in st.session_state:
    st.session_state.test_df = None

# --- タブ構成 ---
tab1, tab2, tab3, tab4 = st.tabs(["1. 読み込み・軽量化", "2. グラフ可視化 (EDA)", "3. 前処理", "4. 機械学習 (LightGBM)"])

# ==========================================
# TAB 1: データ読み込み・軽量化
# ==========================================
with tab1:
    st.subheader("データのアップロード（CSV / Parquet対応）")
    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader("学習用 / 分析用データ (train)", type=["csv", "parquet"])
        if train_file:
            df, start_m, end_m = load_file(train_file)
            if st.session_state.train_df is None or st.button("データを初期状態にリセット"):
                st.session_state.train_df = df.copy()
            st.success(f"読み込み完了: {df.shape[0]:,}行 × {df.shape[1]}列 | メモリ: {start_m:.2f}MB → {end_m:.2f}MB")

    with col2:
        if mode == "コンペ用モード":
            test_file = st.file_uploader("評価用データ (test)", type=["csv", "parquet"])
            if test_file:
                t_df, t_start, t_end = load_file(test_file)
                if st.session_state.test_df is None:
                    st.session_state.test_df = t_df.copy()
                st.success(f"Test完了: {t_df.shape[0]:,}行 × {t_df.shape[1]}列 | メモリ: {t_start:.2f}MB → {t_end:.2f}MB")

    if st.session_state.train_df is not None:
        st.dataframe(st.session_state.train_df.head(10), use_container_width=True)
        # Parquetダウンロード（次回からの読み込み高速化用）
        parquet_bytes = st.session_state.train_df.to_parquet(index=False)
        st.download_button("軽量化データをParquetで保存", data=parquet_bytes, file_name="train_light.parquet")

# ==========================================
# TAB 2: グラフ可視化 (Plotly)
# ==========================================
with tab2:
    if st.session_state.train_df is None:
        st.info("先にタブ1でデータを読み込んでください。")
    else:
        df = st.session_state.train_df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()

        chart_type = st.selectbox("グラフの種類", ["ヒストグラム（分布）", "箱ひげ図（外れ値確認）", "散布図（2変数の関係）", "相関ヒートマップ"])

        if chart_type == "ヒストグラム（分布）":
            col = st.selectbox("対象列", all_cols)
            fig = px.histogram(df, x=col, marginal="box", title=f"{col} の分布")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "箱ひげ図（外れ値確認）":
            y_col = st.selectbox("数値列 (Y)", num_cols)
            x_col = st.selectbox("カテゴリ列 (X・任意)", ["なし"] + all_cols)
            fig = px.box(df, x=None if x_col == "なし" else x_col, y=y_col, title=f"{y_col} の箱ひげ図")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "散布図（2変数の関係）":
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X軸", num_cols, index=0)
            y_col = c2.selectbox("Y軸", num_cols, index=min(1, len(num_cols)-1))
            color_col = c3.selectbox("色分け（任意）", ["なし"] + all_cols)
            # 重くならないよう5000行以上はサンプリング表示
            plot_df = df.sample(5000, random_state=42) if len(df) > 5000 else df
            fig = px.scatter(plot_df, x=x_col, y=y_col, color=None if color_col == "なし" else color_col, opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "相関ヒートマップ":
            if len(num_cols) > 1:
                corr = df[num_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 3: 前処理
# ==========================================
with tab3:
    if st.session_state.train_df is None:
        st.info("先にタブ1でデータを読み込んでください。")
    else:
        df = st.session_state.train_df
        st.write("### 欠損値の確認")
        null_df = pd.DataFrame({"欠損数": df.isnull().sum(), "欠損率(%)": (df.isnull().mean() * 100).round(2), "型": df.dtypes.astype(str)})
        st.dataframe(null_df[null_df["欠損数"] > 0], use_container_width=True)

        st.write("### 基本前処理の実行")
        drop_cols = st.multiselect("削除する列（ID列など）", df.columns.tolist())
        fill_method = st.selectbox("数値列の欠損補完", ["何もしない（LightGBMに任せる）", "中央値で補完", "平均値で補完", "0で補完"])

        if st.button("前処理を適用する"):
            if drop_cols:
                st.session_state.train_df = st.session_state.train_df.drop(columns=drop_cols, errors="ignore")
                if st.session_state.test_df is not None:
                    st.session_state.test_df = st.session_state.test_df.drop(columns=drop_cols, errors="ignore")

            if fill_method != "何もしない（LightGBMに任せる）":
                num_c = st.session_state.train_df.select_dtypes(include=[np.number]).columns
                for c in num_c:
                    val = st.session_state.train_df[c].median() if "中央値" in fill_method else (st.session_state.train_df[c].mean() if "平均値" in fill_method else 0)
                    st.session_state.train_df[c] = st.session_state.train_df[c].fillna(val)
                    if st.session_state.test_df is not None and c in st.session_state.test_df.columns:
                        st.session_state.test_df[c] = st.session_state.test_df[c].fillna(val)
            st.success("前処理を適用しました！")

# ==========================================
# TAB 4: 機械学習 (LightGBM + CV)
# ==========================================
with tab4:
    if st.session_state.train_df is None:
        st.info("先にタブ1でデータを読み込んでください。")
    else:
        df = st.session_state.train_df
        c1, c2, c3 = st.columns(3)
        target_col = c1.selectbox("目的変数 (Target)", df.columns.tolist(), index=len(df.columns)-1)
        task_type = c2.selectbox("タスク種別", ["回帰 (RMSE)", "二値分類 (AUC)"])
        n_splits = c3.slider("CV分割数 (K-Fold)", 3, 10, 5)

        if st.button("🚀 LightGBMで学習・CV実行"):
            X = df.drop(columns=[target_col]).copy()
            y = df[target_col].copy()

            # object/string型をcategory型へ変換（LightGBMネイティブ対応）
            for col in X.select_dtypes(include=["object", "string"]).columns:
                X[col] = X[col].astype("category")

            oof_preds = np.zeros(len(df))
            importances = np.zeros(X.shape[1])

            if mode == "コンペ用モード" and st.session_state.test_df is not None:
                X_test = st.session_state.test_df[X.columns].copy()
                for col in X_test.select_dtypes(include=["object", "string"]).columns:
                    X_test[col] = X_test[col].astype("category")
                test_preds = np.zeros(len(X_test))
            else:
                X_test, test_preds = None, None

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) if "回帰" in task_type else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            progress = st.progress(0)
            for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

                if "回帰" in task_type:
                    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42 + fold, n_jobs=-1)
                else:
                    model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42 + fold, n_jobs=-1)

                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )

                if "回帰" in task_type:
                    oof_preds[va_idx] = model.predict(X_va)
                    if X_test is not None:
                        test_preds += model.predict(X_test) / n_splits
                else:
                    oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
                    if X_test is not None:
                        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

                importances += model.feature_importances_ / n_splits
                progress.progress((fold + 1) / n_splits)

            # スコア表示
            if "回帰" in task_type:
                score = root_mean_squared_error(y, oof_preds)
                st.metric("OOF RMSE（低いほど良い）", f"{score:.5f}")
            else:
                score = roc_auc_score(y, oof_preds)
                st.metric("OOF ROC-AUC（高いほど良い）", f"{score:.5f}")

            # 特徴量重要度グラフ
            imp_df = pd.DataFrame({"特徴量": X.columns, "重要度": importances}).sort_values("重要度", ascending=True).tail(20)
            fig_imp = px.bar(imp_df, x="重要度", y="特徴量", orientation="h", title="特徴量重要度 (Top 20)")
            st.plotly_chart(fig_imp, use_container_width=True)

            # コンペ用：提出ファイル出力
            if test_preds is not None:
                sub_df = pd.DataFrame({"prediction": test_preds})
                st.download_button(
                    "📥 予測結果 (submission.csv) をダウンロード",
                    data=sub_df.to_csv(index=False).encode("utf-8"),
                    file_name="submission.csv",
                    mime="text/csv"
                )
