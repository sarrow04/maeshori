import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_log_error,
    roc_auc_score,
    accuracy_score,
    f1_score,
    log_loss
)

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
    "5. 機械学習・評価"
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

        chart_type = st.selectbox(
            "グラフの種類",
            ["ヒストグラム（分布）", "時系列折れ線グラフ", "箱ひげ図（外れ値確認）", "散布図（2変数の関係）", "相関ヒートマップ"]
        )

        if chart_type == "ヒストグラム（分布）":
            col = st.selectbox("対象列", all_cols)
            fig = px.histogram(df, x=col, marginal="box", title=f"{col} の分布")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "時系列折れ線グラフ":
            if len(num_cols) == 0:
                st.warning("数値列がありません。")
            else:
                c1, c2, c3 = st.columns(3)
                x_col = c1.selectbox("時間軸 (X)", all_cols)
                y_col = c2.selectbox("数値列 (Y)", num_cols)
                color_col = c3.selectbox("グループ色分け（任意）", ["なし"] + all_cols)
                plot_df = df.sort_values(x_col)
                if len(plot_df) > 10000:
                    st.caption("※描画軽量化のため先頭10,000件を表示しています")
                    plot_df = plot_df.head(10000)
                fig = px.line(plot_df, x=x_col, y=y_col, color=None if color_col == "なし" else color_col, title=f"{x_col} ごとの {y_col} 推移")
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

                    if ts_sort_col != "並び替えない（現在の行順）":
                        sort_keys = [ts_grp_col, ts_sort_col] if ts_grp_col != "なし（全体で計算）" else [ts_sort_col]
                        work = work.sort_values(sort_keys).reset_index(drop=True)

                    created_cols = []
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

                    if use_both:
                        new_tr = work[work["_is_train"] == 1].sort_values("_orig_idx").drop(columns=["_orig_idx", "_is_train"]).reset_index(drop=True)
                        new_te = work[work["_is_train"] == 0].sort_values("_orig_idx").drop(columns=["_orig_idx", "_is_train"]).reset_index(drop=True)
                        te_keep = list(st.session_state.test_df.columns) + created_cols
                        st.session_state.train_df = new_tr
                        st.session_state.test_df = new_te[[c for c in te_keep if c in new_te.columns]]
                    else:
                        st.session_state.train_df = work.sort_values("_orig_idx").drop(columns=["_orig_idx"]).reset_index(drop=True)

                    st.success(f"時系列特徴量を {len(created_cols)} 列追加しました！: {', '.join(created_cols)}")

# ==========================================
# TAB 5: 機械学習・評価
# ==========================================
with tab5:
    if st.session_state.train_df is None:
        st.info("先にタブ1でデータを読み込んでください。")
    else:
        df = st.session_state.train_df
        all_cols = df.columns.tolist()

        # --- 基本設定 ---
        c1, c2, c3 = st.columns(3)
        target_col = c1.selectbox("目的変数 (Target)", all_cols, index=len(all_cols)-1)
        task_type = c2.selectbox("タスク種別", ["回帰", "二値分類"])
        model_name = c3.selectbox(
            "使用モデル",
            ["LightGBM", "決定木 (DecisionTree)", "ランダムフォレスト (RandomForest)", "線形モデル (Ridge / Logistic)"]
        )

        # --- 使用する特徴量カラムの選択 ---
        default_features = [c for c in all_cols if c != target_col]
        feature_cols = st.multiselect("学習に使う特徴量カラムを選択", default_features, default=default_features)

        # --- CV・評価指標設定 ---
        c4, c5, c6 = st.columns(3)
        cv_type = c4.selectbox("CV（分割）方法", ["K-Fold / Stratified K-Fold（通常）", "TimeSeriesSplit（時系列分析用）"])
        n_splits = c5.slider("CV分割数", 3, 10, 5)

        if task_type == "回帰":
            use_log1p = c6.checkbox("目的変数を log1p 変換（RMSLE最適化）", value=False)
        else:
            use_log1p = False
            c6.write("")

        ts_cv_sort_col = None
        if "TimeSeriesSplit" in cv_type:
            ts_cv_sort_col = st.selectbox("時系列CV用の並び替え列（日付・時刻など）", ["並び替えない（現在の行順）"] + all_cols)

        # モデル別の簡易パラメータ
        max_depth = None
        if model_name in ["決定木 (DecisionTree)", "ランダムフォレスト (RandomForest)"]:
            max_depth = st.slider("木の最大深さ (max_depth)", 2, 20, 6)

        if st.button("🚀 学習・CV評価を実行"):
            if not feature_cols:
                st.error("学習に使う特徴量カラムを1つ以上選択してください。")
                st.stop()

            try:
                work_df = df.dropna(subset=[target_col]).copy()
                if "TimeSeriesSplit" in cv_type and ts_cv_sort_col and ts_cv_sort_col != "並び替えない（現在の行順）":
                    work_df = work_df.sort_values(ts_cv_sort_col)
                work_df = work_df.reset_index(drop=True)

                X = work_df[feature_cols].copy()
                y = work_df[target_col].copy()

                # 二値分類チェック
                if task_type == "二値分類":
                    if y.nunique() != 2:
                        st.error(f"「二値分類」が選ばれていますが、目的変数 '{target_col}' のユニーク数が {y.nunique()} 個です。")
                        st.stop()
                    if not pd.api.types.is_numeric_dtype(y):
                        y = pd.Series(pd.factorize(y)[0])

                # RMSLE (log1p) チェック
                if use_log1p and (y < 0).any():
                    st.error("目的変数に負の値が含まれているため、log1p変換（RMSLE最適化）は使用できません。")
                    st.stop()

                # テストデータ準備
                if mode == "コンペ用モード" and st.session_state.test_df is not None:
                    missing_in_test = [c for c in feature_cols if c not in st.session_state.test_df.columns]
                    if missing_in_test:
                        st.error(f"testデータに以下の選択カラムが存在しません: {missing_in_test}")
                        st.stop()
                    X_test = st.session_state.test_df[feature_cols].copy()
                else:
                    X_test = None

                # モデル別の型処理
                if model_name == "LightGBM":
                    for col in X.select_dtypes(include=["object", "string"]).columns:
                        X[col] = X[col].astype("category")
                        if X_test is not None:
                            X_test[col] = X_test[col].astype("category")
                else:
                    # 決定木・RF・線形モデルはカテゴリを数値化＆欠損を中央値補完
                    for col in X.columns:
                        if not pd.api.types.is_numeric_dtype(X[col]):
                            cat_type = X[col].astype("category")
                            X[col] = cat_type.cat.codes
                            if X_test is not None:
                                X_test[col] = pd.Categorical(X_test[col], categories=cat_type.cat.categories).codes
                        med = X[col].median()
                        med = 0 if pd.isna(med) else med
                        X[col] = X[col].fillna(med)
                        if X_test is not None:
                            X_test[col] = X_test[col].fillna(med)

                # CV分割器
                if "TimeSeriesSplit" in cv_type:
                    splitter = TimeSeriesSplit(n_splits=n_splits)
                    splits = list(splitter.split(X))
                elif task_type == "回帰":
                    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    splits = list(splitter.split(X))
                else:
                    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    splits = list(splitter.split(X, y))

                oof_preds = np.full(len(work_df), np.nan)
                test_preds = np.zeros(len(X_test)) if X_test is not None else None
                importances = np.zeros(len(feature_cols))

                progress = st.progress(0)
                for fold, (tr_idx, va_idx) in enumerate(splits):
                    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

                    y_tr_fit = np.log1p(y_tr) if use_log1p else y_tr
                    y_va_fit = np.log1p(y_va) if use_log1p else y_va

                    # モデル定義
                    if model_name == "LightGBM":
                        if task_type == "回帰":
                            model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42 + fold, n_jobs=-1)
                        else:
                            model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42 + fold, n_jobs=-1)
                        model.fit(X_tr, y_tr_fit, eval_set=[(X_va, y_va_fit)], callbacks=[lgb.early_stopping(50, verbose=False)])
                    elif model_name == "決定木 (DecisionTree)":
                        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42 + fold) if task_type == "回帰" else DecisionTreeClassifier(max_depth=max_depth, random_state=42 + fold)
                        model.fit(X_tr, y_tr_fit)
                    elif model_name == "ランダムフォレスト (RandomForest)":
                        model = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=42 + fold, n_jobs=-1) if task_type == "回帰" else RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=42 + fold, n_jobs=-1)
                        model.fit(X_tr, y_tr_fit)
                    else:
                        if task_type == "回帰":
                            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                        else:
                            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                        model.fit(X_tr, y_tr_fit)

                    # 予測
                    if task_type == "回帰":
                        va_pred = model.predict(X_va)
                        if use_log1p:
                            va_pred = np.expm1(np.clip(va_pred, -10, 50))
                        oof_preds[va_idx] = va_pred
                        if X_test is not None:
                            t_pred = model.predict(X_test)
                            if use_log1p:
                                t_pred = np.expm1(np.clip(t_pred, -10, 50))
                            test_preds += t_pred / len(splits)
                    else:
                        oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
                        if X_test is not None:
                            test_preds += model.predict_proba(X_test)[:, 1] / len(splits)

                    # 重要度取得
                    if hasattr(model, "feature_importances_"):
                        importances += model.feature_importances_ / len(splits)
                    elif hasattr(model, "named_steps"):
                        est = list(model.named_steps.values())[-1]
                        coef = np.abs(est.coef_).ravel()
                        importances += coef / len(splits)

                    progress.progress((fold + 1) / len(splits))

                # --- 評価指標の一覧表示 ---
                valid_mask = ~np.isnan(oof_preds)
                y_eval = y[valid_mask]
                p_eval = oof_preds[valid_mask]

                st.write("### 📈 評価スコア (OOF)")
                if task_type == "回帰":
                    rmse = root_mean_squared_error(y_eval, p_eval)
                    mae = mean_absolute_error(y_eval, p_eval)
                    r2 = r2_score(y_eval, p_eval)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("RMSE（低いほど良い）", f"{rmse:.5f}")
                    m2.metric("MAE（低いほど良い）", f"{mae:.5f}")
                    m3.metric("R²（1に近いほど良い）", f"{r2:.5f}")
                    if (y_eval >= 0).all():
                        rmsle = root_mean_squared_log_error(y_eval, np.clip(p_eval, 0, None))
                        m4.metric("RMSLE（低いほど良い）", f"{rmsle:.5f}")
                    else:
                        m4.metric("RMSLE", "負の値あり算出不可")
                else:
                    auc = roc_auc_score(y_eval, p_eval)
                    pred_cls = (p_eval >= 0.5).astype(int)
                    acc = accuracy_score(y_eval, pred_cls)
                    f1 = f1_score(y_eval, pred_cls, zero_division=0)
                    ll = log_loss(y_eval, np.clip(p_eval, 1e-15, 1 - 1e-15))
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("ROC-AUC（高いほど良い）", f"{auc:.5f}")
                    m2.metric("Accuracy", f"{acc:.5f}")
                    m3.metric("F1 Score", f"{f1:.5f}")
                    m4.metric("LogLoss（低いほど良い）", f"{ll:.5f}")

                # --- 特徴量重要度 ---
                imp_title = "特徴量重要度 (Top 20)" if model_name != "線形モデル (Ridge / Logistic)" else "回帰係数の絶対値 (Top 20)"
                imp_df = pd.DataFrame({"特徴量": feature_cols, "重要度": importances}).sort_values("重要度", ascending=True).tail(20)
                fig_imp = px.bar(imp_df, x="重要度", y="特徴量", orientation="h", title=imp_title)
                st.plotly_chart(fig_imp, use_container_width=True)

                # --- コンペ用提出ファイル ---
                if test_preds is not None:
                    sub_df = pd.DataFrame({"prediction": test_preds})
                    st.download_button(
                        "📥 予測結果 (submission.csv) をダウンロード",
                        data=sub_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"submission_{model_name.split()[0]}.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"学習中にエラーが発生しました: {e}")
