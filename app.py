import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import root_mean_squared_error, roc_auc_score

st.set_page_config(page_title="Data Analysis & Competition App", layout="wide")

# --- メモリ軽量化関数 ---
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if df[col].isnull().all() or pd.api.types.is_bool_dtype(df[col]):
            continue

        if pd.api.types.is_integer_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif pd.api.types.is_float_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

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

if "train_df" not in st.session_state:
    st.session_state.train_df = None
if "test_df" not in st.session_state:
    st.session_state.test_df = None

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. 読み込み・概要",
    "2. グラフ可視化 (EDA)",
    "3. 前処理",
    "4. 特徴量作成",
    "5. 機械学習 (LightGBM)"
])

# ==========================================
# TAB 1: データ読み込み・軽量化・ユニーク確認
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
        df_show = st.session_state.train_df

        st.write("### カラム概要（型・欠損・ユニーク数）")
        top_vals = []
        for c in df_show.columns:
            vc = df_show[c].value_counts(dropna=False).head(3)
            top_vals.append(", ".join([f"{k}({v:,})" for k, v in vc.items()]))

        summary_df = pd.DataFrame({
            "型": df_show.dtypes.astype(str),
            "ユニーク数": df_show.nunique(),
            "ユニーク率(%)": (df_show.nunique() / len(df_show) * 100).round(2),
            "欠損数": df_show.isnull().sum(),
            "欠損率(%)": (df_show.isnull().mean() * 100).round(2),
            "頻出トップ3（値(出現数)）": top_vals
        })
        st.dataframe(summary_df, use_container_width=True)

        st.write("### カラム別の全ユニーク値・出現数")
        selected_col = st.selectbox("詳細を見たいカラムを選択", df_show.columns.tolist())
        if selected_col:
            vc_df = df_show[selected_col].value_counts(dropna=False).reset_index()
            vc_df.columns = ["値（NaN含む）", "出現数"]
            vc_df["割合(%)"] = (vc_df["出現数"] / len(df_show) * 100).round(2)
            st.caption(f"「{selected_col}」の全ユニーク値: {len(vc_df):,}種類（出現数が多い順）")
            st.dataframe(vc_df, use_container_width=True)

        st.write("### データプレビュー（先頭10行）")
        st.dataframe(df_show.head(10), use_container_width=True)

        parquet_bytes = df_show.to_parquet(index=False)
        st.download_button("軽量化・加工済みデータをParquetで保存", data=parquet_bytes, file_name="train_processed.parquet")

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
            if len(num_cols) == 0:
                st.warning("数値列がありません。")
            else:
                y_col = st.selectbox("数値列 (Y)", num_cols)
                x_col = st.selectbox("カテゴリ列 (X・任意)", ["なし"] + all_cols)
                fig = px.box(df, x=None if x_col == "なし" else x_col, y=y_col, title=f"{y_col} の箱ひげ図")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "散布図（2変数の関係）":
            if len(num_cols) < 2:
                st.warning("数値列が2つ以上必要です。")
            else:
                c1, c2, c3 = st.columns(3)
                x_col = c1.selectbox("X軸", num_cols, index=0)
                y_col = c2.selectbox("Y軸", num_cols, index=1)
                color_col = c3.selectbox("色分け（任意）", ["なし"] + all_cols)
                plot_df = df.sample(5000, random_state=42) if len(df) > 5000 else df
                fig = px.scatter(plot_df, x=x_col, y=y_col, color=None if color_col == "なし" else color_col, opacity=0.6)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "相関ヒートマップ":
            if len(num_cols) > 1:
                corr = df[num_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("数値列が2つ以上必要です。")

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
# TAB 4: 特徴量エンジニアリング
# ==========================================
with tab4:
    if st.session_state.train_df is None:
        st.info("先にタブ1でデータを読み込んでください。")
    else:
        df = st.session_state.train_df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()

        fe_type = st.radio(
            "作成する特徴量の種類",
            ["1. 四則演算（比率・差分など）", "2. GroupBy集約（カテゴリ別集計）", "3. 時系列（ラグ・差分・移動平均）"],
            horizontal=True
        )

        # --- 1. 四則演算 ---
        if fe_type == "1. 四則演算（比率・差分など）":
            if len(num_cols) < 2:
                st.warning("数値列が2つ以上必要です。")
            else:
                c1, c2, c3 = st.columns(3)
                col_a = c1.selectbox("列 A", num_cols, key="fe_a")
                op = c2.selectbox("演算", ["割り算 (A / B)", "引き算 (A - B)", "掛け算 (A * B)", "足し算 (A + B)"])
                col_b = c3.selectbox("列 B", num_cols, index=min(1, len(num_cols)-1), key="fe_b")

                op_map = {"割り算 (A / B)": "div", "引き算 (A - B)": "sub", "掛け算 (A * B)": "mul", "足し算 (A + B)": "add"}
                new_col_name = st.text_input("新しい列名", value=f"{col_a}_{op_map[op]}_{col_b}")

                if st.button("四則演算の特徴量を追加"):
                    def apply_op(target_df):
                        if col_a in target_df.columns and col_b in target_df.columns:
                            if "割り算" in op:
                                target_df[new_col_name] = np.where(target_df[col_b] == 0, np.nan, target_df[col_a] / target_df[col_b]).astype(np.float32)
                            elif "引き算" in op:
                                target_df[new_col_name] = (target_df[col_a] - target_df[col_b]).astype(np.float32)
                            elif "掛け算" in op:
                                target_df[new_col_name] = (target_df[col_a] * target_df[col_b]).astype(np.float32)
                            else:
                                target_df[new_col_name] = (target_df[col_a] + target_df[col_b]).astype(np.float32)
                        return target_df

                    st.session_state.train_df = apply_op(st.session_state.train_df)
                    if st.session_state.test_df is not None:
                        st.session_state.test_df = apply_op(st.session_state.test_df)
                    st.success(f"特徴量「{new_col_name}」を追加しました！")

        # --- 2. GroupBy集約 ---
        elif fe_type == "2. GroupBy集約（カテゴリ別集計）":
            if len(num_cols) == 0:
                st.warning("数値列がありません。")
            else:
                c1, c2 = st.columns(2)
                grp_col = c1.selectbox("グループ化する列（カテゴリ/IDなど）", all_cols, key="gb_grp")
                val_col = c2.selectbox("集計する数値列", num_cols, key="gb_val")
                agg_funcs = st.multiselect("集計方法", ["mean", "max", "min", "std", "median", "count"], default=["mean", "std"])

                if st.button("GroupBy特徴量を追加"):
                    if not agg_funcs:
                        st.warning("集計方法を1つ以上選んでください。")
                    else:
                        tr = st.session_state.train_df
                        te = st.session_state.test_df

                        # trainとtestの両方に列があれば結合して一括集計
                        use_both = (te is not None) and (grp_col in te.columns) and (val_col in te.columns)
                        if use_both:
                            tr_len = len(tr)
                            comb = pd.concat([tr[[grp_col, val_col]], te[[grp_col, val_col]]], ignore_index=True)
                            for func in agg_funcs:
                                new_name = f"{grp_col}_{val_col}_{func}"
                                vals = comb.groupby(grp_col, observed=False)[val_col].transform(func).astype(np.float32)
                                tr[new_name] = vals.iloc[:tr_len].values
                                te[new_name] = vals.iloc[tr_len:].values
                            st.session_state.train_df = tr
                            st.session_state.test_df = te
                        else:
                            for func in agg_funcs:
                                new_name = f"{grp_col}_{val_col}_{func}"
                                tr[new_name] = tr.groupby(grp_col, observed=False)[val_col].transform(func).astype(np.float32)
                            st.session_state.train_df = tr

                        st.success(f"GroupBy特徴量（{len(agg_funcs)}列）を追加しました！")

        # --- 3. 時系列（ラグ・差分・移動平均） ---
        elif fe_type == "3. 時系列（ラグ・差分・移動平均）":
            if len(num_cols) == 0:
                st.warning("数値列がありません。")
            else:
                c1, c2, c3 = st.columns(3)
                ts_val_col = c1.selectbox("対象の数値列", num_cols, key="ts_val")
                ts_sort_col = c2.selectbox("並び替え列（日付・時刻など）", ["並び替えない（現在の行順）"] + all_cols, key="ts_sort")
                ts_grp_col = c3.selectbox("グループ列（店舗IDなど・任意）", ["なし（全体で計算）"] + all_cols, key="ts_grp")

                c4, c5 = st.columns(2)
                lags = c4.multiselect("作成するラグ (shift)", [1, 2, 3, 7, 14, 28, 30], default=[1, 7])
                windows = c5.multiselect("移動平均・標準偏差の窓幅 (rolling)", [3, 7, 14, 28, 30], default=[7])

                c6, c7 = st.columns(2)
                add_diff = c6.checkbox("ラグとの差分 (diff) も作成する", value=True)
                prevent_leak = c7.checkbox("移動平均を1期シフトしてリークを防ぐ（推奨）", value=True)

                if st.button("時系列特徴量を追加"):
                    tr = st.session_state.train_df.copy()
                    te = st.session_state.test_df.copy() if st.session_state.test_df is not None else None

                    # trainとtestを結合して時系列計算するかどうか判定
                    req_cols = [ts_val_col]
                    if ts_sort_col != "並び替えない（現在の行順）":
                        req_cols.append(ts_sort_col)
                    if ts_grp_col != "なし（全体で計算）":
                        req_cols.append(ts_grp_col)

                    use_both = (te is not None) and all(c in te.columns for c in req_cols)

                    if use_both:
                        tr["_orig_idx"] = np.arange(len(tr))
                        tr["_is_train"] = 1
                        te["_orig_idx"] = np.arange(len(te))
                        te["_is_train"] = 0
                        work = pd.concat([tr, te], ignore_index=True)
                    else:
                        tr["_orig_idx"] = np.arange(len(tr))
                        work = tr

                    # ソート
                    if ts_sort_col != "並び替えない（現在の行順）":
                        sort_keys = [ts_grp_col, ts_sort_col] if ts_grp_col != "なし（全体で計算）" else [ts_sort_col]
                        work = work.sort_values(sort_keys).reset_index(drop=True)

                    created_cols = []
                    # ラグ・差分
                    for lag in lags:
                        lag_col = f"{ts_val_col}_lag{lag}"
                        if ts_grp_col != "なし（全体で計算）":
                            work[lag_col] = work.groupby(ts_grp_col, observed=False)[ts_val_col].shift(lag).astype(np.float32)
                        else:
                            work[lag_col] = work[ts_val_col].shift(lag).astype(np.float32)
                        created_cols.append(lag_col)

                        if add_diff:
                            diff_col = f"{ts_val_col}_diff{lag}"
                            work[diff_col] = (work[ts_val_col] - work[lag_col]).astype(np.float32)
                            created_cols.append(diff_col)

                    # 移動平均・移動標準偏差
                    shift_n = 1 if prevent_leak else 0
                    for w in windows:
                        ma_col = f"{ts_val_col}_ma{w}"
                        std_col = f"{ts_val_col}_mstd{w}"
                        if ts_grp_col != "なし（全体で計算）":
                            s = work.groupby(ts_grp_col, observed=False)[ts_val_col].shift(shift_n)
                            work[ma_col] = s.groupby(work[ts_grp_col], observed=False).transform(lambda x: x.rolling(w, min_periods=1).mean()).astype(np.float32)
                            work[std_col] = s.groupby(work[ts_grp_col], observed=False).transform(lambda x: x.rolling(w, min_periods=1).std()).astype(np.float32)
                        else:
                            s = work[ts_val_col].shift(shift_n)
                            work[ma_col] = s.rolling(w, min_periods=1).mean().astype(np.float32)
                            work[std_col] = s.rolling(w, min_periods=1).std().astype(np.float32)
                        created_cols.extend([ma_col, std_col])

                    # 元の行順・train/testに戻す
                    if use_both:
                        new_tr = work[work["_is_train"] == 1].sort_values("_orig_idx").drop(columns=["_orig_idx", "_is_train"]).reset_index(drop=True)
                        new_te = work[work["_is_train"] == 0].sort_values("_orig_idx").drop(columns=["_orig_idx", "_is_train"]).reset_index(drop=True)
                        # testに元々なかった列（目的変数など）を除外
                        te_keep = list(st.session_state.test_df.columns) + created_cols
                        st.session_state.train_df = new_tr
                        st.session_state.test_df = new_te[[c for c in te_keep if c in new_te.columns]]
                    else:
                        st.session_state.train_df = work.sort_values("_orig_idx").drop(columns=["_orig_idx"]).reset_index(drop=True)

                    st.success(f"時系列特徴量を {len(created_cols)} 列追加しました！: {', '.join(created_cols)}")

# ==========================================
# TAB 5: 機械学習 (LightGBM + CV)
# ==========================================
with tab5:
    if st.session_state.train_df is None:
        st.info("先にタブ1でデータを読み込んでください。")
    else:
        df = st.session_state.train_df
        c1, c2, c3 = st.columns(3)
        target_col = c1.selectbox("目的変数 (Target)", df.columns.tolist(), index=len(df.columns)-1)
        task_type = c2.selectbox("タスク種別", ["回帰 (RMSE)", "二値分類 (AUC)"])
        n_splits = c3.slider("CV分割数 (K-Fold)", 3, 10, 5)

        if st.button("🚀 LightGBMで学習・CV実行"):
            try:
                work_df = df.dropna(subset=[target_col]).reset_index(drop=True)
                X = work_df.drop(columns=[target_col]).copy()
                y = work_df[target_col].copy()

                if "二値分類" in task_type:
                    n_unique = y.nunique()
                    if n_unique != 2:
                        st.error(f"「二値分類」が選ばれていますが、目的変数 '{target_col}' のユニーク数が {n_unique} 個です（2個である必要があります）。回帰タスクの場合は「回帰 (RMSE)」を選んでください。")
                        st.stop()
                    if not pd.api.types.is_numeric_dtype(y):
                        y = pd.Series(pd.factorize(y)[0])

                for col in X.select_dtypes(include=["object", "string"]).columns:
                    X[col] = X[col].astype("category")

                oof_preds = np.zeros(len(work_df))
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

                if "回帰" in task_type:
                    score = root_mean_squared_error(y, oof_preds)
                    st.metric("OOF RMSE（低いほど良い）", f"{score:.5f}")
                else:
                    score = roc_auc_score(y, oof_preds)
                    st.metric("OOF ROC-AUC（高いほど良い）", f"{score:.5f}")

                imp_df = pd.DataFrame({"特徴量": X.columns, "重要度": importances}).sort_values("重要度", ascending=True).tail(20)
                fig_imp = px.bar(imp_df, x="重要度", y="特徴量", orientation="h", title="特徴量重要度 (Top 20)")
                st.plotly_chart(fig_imp, use_container_width=True)

                if test_preds is not None:
                    sub_df = pd.DataFrame({"prediction": test_preds})
                    st.download_button(
                        "📥 予測結果 (submission.csv) をダウンロード",
                        data=sub_df.to_csv(index=False).encode("utf-8"),
                        file_name="submission.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"学習中にエラーが発生しました: {e}")
