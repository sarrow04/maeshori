import os
import io
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import lightgbm as lgb
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
    train_test_split
)
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

# --- 自動保存（バックアップ）用パス ---
AUTOSAVE_DIR = ".autosave_cache"
os.makedirs(AUTOSAVE_DIR, exist_ok=True)
TRAIN_CACHE_PATH = os.path.join(AUTOSAVE_DIR, "train_backup.parquet")
TEST_CACHE_PATH = os.path.join(AUTOSAVE_DIR, "test_backup.parquet")

def save_backup():
    try:
        if st.session_state.train_df is not None:
            st.session_state.train_df.to_parquet(TRAIN_CACHE_PATH, index=False)
        if st.session_state.test_df is not None:
            st.session_state.test_df.to_parquet(TEST_CACHE_PATH, index=False)
    except Exception:
        pass

def notify_and_refresh(msg):
    save_backup()
    st.session_state.flash_msg = msg
    st.rerun()

def get_all_datasets():
    """管理中の全データセットを辞書で返す"""
    datasets = {}
    if st.session_state.train_df is not None:
        datasets["train"] = st.session_state.train_df
    if st.session_state.test_df is not None:
        datasets["test"] = st.session_state.test_df
    for name, df_val in st.session_state.extra_dfs.items():
        datasets[name] = df_val
    return datasets

def render_custom_save_ui(key_prefix):
    st.divider()
    st.write("### 💾 現在のデータを名前を付けて保存")
    c1, c2 = st.columns(2)
    with c1:
        tr_name = st.text_input("Trainのファイル名（拡張子不要）", value="train_processed", key=f"{key_prefix}_tr_name")
        b1, b2 = st.columns(2)
        b1.download_button(
            "📥 Train (CSV)",
            data=st.session_state.train_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{tr_name}.csv",
            mime="text/csv",
            key=f"{key_prefix}_tr_csv"
        )
        b2.download_button(
            "📥 Train (Parquet)",
            data=st.session_state.train_df.to_parquet(index=False),
            file_name=f"{tr_name}.parquet",
            key=f"{key_prefix}_tr_pq"
        )
    with c2:
        if st.session_state.test_df is not None:
            te_name = st.text_input("Testのファイル名（拡張子不要）", value="test_processed", key=f"{key_prefix}_te_name")
            b3, b4 = st.columns(2)
            b3.download_button(
                "📥 Test (CSV)",
                data=st.session_state.test_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{te_name}.csv",
                mime="text/csv",
                key=f"{key_prefix}_te_csv"
            )
            b4.download_button(
                "📥 Test (Parquet)",
                data=st.session_state.test_df.to_parquet(index=False),
                file_name=f"{te_name}.parquet",
                key=f"{key_prefix}_te_pq"
            )

# --- リアルタイム確認パネル（先頭10行 & 基本統計量） ---
def render_live_preview():
    datasets = get_all_datasets()
    if not datasets:
        return
    tr_info = f"Train: {st.session_state.train_df.shape[0]:,}行 × {st.session_state.train_df.shape[1]}列" if st.session_state.train_df is not None else f"登録データ: {len(datasets)}件"
    label = f"⚡ リアルタイム確認：基本統計量 & 10行プレビュー（{tr_info} / 全{len(datasets)}データ）"
    with st.expander(label, expanded=False):
        c1, c2 = st.columns(2)
        view_target = c1.selectbox("表示するデータ", list(datasets.keys()), key="live_view_target")
        view_pos = c2.radio("表示位置（10行）", ["先頭10行", "末尾10行"], horizontal=True, key="live_view_pos")

        target_df = datasets[view_target]
        mem_mb = target_df.memory_usage().sum() / 1024**2
        st.caption(f"サイズ: {target_df.shape[0]:,}行 × {target_df.shape[1]}列 | メモリ: {mem_mb:.2f}MB | 総欠損数: {target_df.isnull().sum().sum():,}")

        st.write("**📄 データプレビュー（10行）**")
        preview_df = target_df.head(10) if view_pos == "先頭10行" else target_df.tail(10)
        st.dataframe(preview_df, use_container_width=True)

        st.write("**📊 基本統計量（数値列）**")
        num_df = target_df.select_dtypes(include=[np.number])
        if num_df.shape[1] > 0:
            desc_df = num_df.describe().T
            desc_df.columns = ["件数", "平均", "標準偏差", "最小", "25%", "中央値(50%)", "75%", "最大"]
            st.dataframe(desc_df.round(3), use_container_width=True)
        else:
            st.info("数値列がありません。")

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

# --- CSV / Parquet / ZIP 共通読み込み関数 ---
def read_uploaded_dataframe(uploaded_file, has_header=True):
    fname = uploaded_file.name.lower()
    if fname.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file) as z:
            targets = [
                f for f in z.namelist()
                if not f.startswith("__MACOSX") and not f.endswith("/")
                and (f.lower().endswith(".parquet") or f.lower().endswith(".csv"))
            ]
            if not targets:
                raise ValueError("ZIP内にCSVまたはParquetファイルが見つかりませんでした。")
            inner_name = targets[0]
            with z.open(inner_name) as f:
                data_bytes = io.BytesIO(f.read())
                if inner_name.lower().endswith(".parquet"):
                    return pd.read_parquet(data_bytes)
                return pd.read_csv(data_bytes, header=0 if has_header else None)
    elif fname.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    else:
        return pd.read_csv(uploaded_file, header=0 if has_header else None)

@st.cache_data
def load_file(uploaded_file):
    df = read_uploaded_dataframe(uploaded_file, has_header=True)
    return reduce_mem_usage(df)

@st.cache_data
def load_sub_file(uploaded_file, has_header):
    return read_uploaded_dataframe(uploaded_file, has_header=has_header)

# --- 行番号文字列（例: "0, 5, 10-15"）のパース関数 ---
def parse_row_indices(text, max_len):
    indices = set()
    for part in text.replace("、", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) == 2 and bounds[0].strip().isdigit() and bounds[1].strip().isdigit():
                start_i, end_i = int(bounds[0].strip()), int(bounds[1].strip())
                for idx in range(min(start_i, end_i), max(start_i, end_i) + 1):
                    if 0 <= idx < max_len:
                        indices.add(idx)
        elif part.isdigit():
            idx = int(part)
            if 0 <= idx < max_len:
                indices.add(idx)
    return sorted(indices)

# --- JIS標準地域メッシュ変換関数 ---
def decode_mesh_code(code):
    if pd.isna(code):
        return np.nan, np.nan
    s = str(code).split(".")[0].strip()
    if not s.isdigit() or len(s) < 4:
        return np.nan, np.nan
    p = int(s[0:2])
    u = int(s[2:4])
    lat = p * 2.0 / 3.0
    lon = u + 100.0
    d_lat, d_lon = 2.0 / 3.0, 1.0
    if len(s) >= 6:
        q, v = int(s[4]), int(s[5])
        d_lat /= 8.0
        d_lon /= 8.0
        lat += q * d_lat
        lon += v * d_lon
    if len(s) >= 8:
        r, w = int(s[6]), int(s[7])
        d_lat /= 10.0
        d_lon /= 10.0
        lat += r * d_lat
        lon += w * d_lon
    if len(s) >= 9:
        m4 = int(s[8])
        d_lat /= 2.0
        d_lon /= 2.0
        lat += ((m4 - 1) // 2) * d_lat
        lon += ((m4 - 1) % 2) * d_lon
    if len(s) >= 10:
        m5 = int(s[9])
        d_lat /= 2.0
        d_lon /= 2.0
        lat += ((m5 - 1) // 2) * d_lat
        lon += ((m5 - 1) % 2) * d_lon
    return lat + d_lat / 2.0, lon + d_lon / 2.0

def encode_latlon_to_mesh(lat_s, lon_s, level=4):
    lat = pd.to_numeric(lat_s, errors="coerce")
    lon = pd.to_numeric(lon_s, errors="coerce")
    valid = lat.notna() & lon.notna()
    res = pd.Series(index=lat.index, dtype="object")
    if not valid.any():
        return res
    la = lat[valid].values
    lo = lon[valid].values
    p = np.floor(la * 1.5).astype(int)
    u = np.floor(lo - 100).astype(int)
    rem_la1 = la * 1.5 - p
    rem_lo1 = (lo - 100) - u
    q = np.floor(rem_la1 * 8).astype(int)
    v = np.floor(rem_lo1 * 8).astype(int)
    rem_la2 = rem_la1 * 8 - q
    rem_lo2 = rem_lo1 * 8 - v
    r = np.floor(rem_la2 * 10).astype(int)
    w = np.floor(rem_lo2 * 10).astype(int)
    if level == 3:
        codes = [f"{a:02d}{b:02d}{c}{d}{e}{f}" for a, b, c, d, e, f in zip(p, u, q, v, r, w)]
    else:
        rem_la3 = rem_la2 * 10 - r
        rem_lo3 = rem_lo2 * 10 - w
        m4 = np.floor(rem_la3 * 2).astype(int) * 2 + np.floor(rem_lo3 * 2).astype(int) + 1
        codes = [f"{a:02d}{b:02d}{c}{d}{e}{f}{g}" for a, b, c, d, e, f, g in zip(p, u, q, v, r, w, m4)]
    res.loc[valid] = codes
    return res

# --- モデル生成関数（チューニングプリセット対応） ---
def build_model(model_name, task_type, preset, seed):
    if model_name == "LightGBM":
        params = {
            "浅い（過学習防止）": dict(n_estimators=1000, learning_rate=0.05, max_depth=4, num_leaves=15, min_child_samples=30, colsample_bytree=0.8, subsample=0.8, subsample_freq=1),
            "標準（バランス）": dict(n_estimators=1000, learning_rate=0.05, max_depth=-1, num_leaves=31, min_child_samples=20, colsample_bytree=0.8, subsample=0.8, subsample_freq=1),
            "深い（表現力重視）": dict(n_estimators=1000, learning_rate=0.05, max_depth=10, num_leaves=63, min_child_samples=10, colsample_bytree=0.8, subsample=0.8, subsample_freq=1),
            "高精度・じっくり（コンペ向け）": dict(n_estimators=2000, learning_rate=0.02, max_depth=7, num_leaves=31, min_child_samples=20, colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_alpha=0.1, reg_lambda=1.0),
        }[preset]
        cls = lgb.LGBMRegressor if task_type == "回帰" else lgb.LGBMClassifier
        return cls(random_state=seed, n_jobs=-1, verbose=-1, **params)

    elif model_name == "決定木 (DecisionTree)":
        params = {
            "浅い（過学習防止）": dict(max_depth=3, min_samples_leaf=10),
            "標準（バランス）": dict(max_depth=6, min_samples_leaf=5),
            "深い（表現力重視）": dict(max_depth=12, min_samples_leaf=2),
            "高精度・じっくり（コンペ向け）": dict(max_depth=8, min_samples_leaf=5),
        }[preset]
        cls = DecisionTreeRegressor if task_type == "回帰" else DecisionTreeClassifier
        return cls(random_state=seed, **params)

    elif model_name == "ランダムフォレスト (RandomForest)":
        params = {
            "浅い（過学習防止）": dict(n_estimators=100, max_depth=4, min_samples_leaf=5),
            "標準（バランス）": dict(n_estimators=100, max_depth=8, min_samples_leaf=2),
            "深い（表現力重視）": dict(n_estimators=150, max_depth=15, min_samples_leaf=1),
            "高精度・じっくり（コンペ向け）": dict(n_estimators=300, max_depth=12, min_samples_leaf=2),
        }[preset]
        cls = RandomForestRegressor if task_type == "回帰" else RandomForestClassifier
        return cls(random_state=seed, n_jobs=-1, **params)

    else:
        if task_type == "回帰":
            alpha = {"浅い（過学習防止）": 10.0, "標準（バランス）": 1.0, "深い（表現力重視）": 0.1, "高精度・じっくり（コンペ向け）": 0.5}[preset]
            return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        else:
            c_val = {"浅い（過学習防止）": 0.1, "標準（バランス）": 1.0, "深い（表現力重視）": 10.0, "高精度・じっくり（コンペ向け）": 2.0}[preset]
            return make_pipeline(StandardScaler(), LogisticRegression(C=c_val, max_iter=1000))

# --- セッション状態の初期化 ---
for key, default_val in [
    ("train_df", None),
    ("test_df", None),
    ("test_raw_df", None),
    ("extra_dfs", {}),
    ("loaded_normal_files", set()),
    ("sub_template_df", None),
    ("sub_has_header", True),
    ("test_preds", None),
    ("last_model_name", ""),
    ("last_task_type", ""),
    ("flash_msg", "")
]:
    if key not in st.session_state:
        st.session_state[key] = default_val

# --- サイドバー：モード切替＆自動バックアップ復元 ---
st.sidebar.title("⚙️ アプリ設定")
mode = st.sidebar.radio("モードを選択", ["通常分析モード", "コンペ用モード"])

st.sidebar.divider()
st.sidebar.subheader("🛡️ データ消失対策（自動保存）")
if os.path.exists(TRAIN_CACHE_PATH):
    if st.sidebar.button("🔄 前回の作業データを復元"):
        st.session_state.train_df = pd.read_parquet(TRAIN_CACHE_PATH)
        if os.path.exists(TEST_CACHE_PATH):
            st.session_state.test_df = pd.read_parquet(TEST_CACHE_PATH)
            st.session_state.test_raw_df = st.session_state.test_df.copy()
        st.session_state.flash_msg = "前回の作業データを復元しました！"
        st.rerun()
else:
    st.sidebar.caption("データを読み込み・加工すると自動でバックアップされます。")

st.title(f"📊 データ分析・MLアプリ（{mode}）")

# 処理完了メッセージの表示
if st.session_state.flash_msg:
    st.success(st.session_state.flash_msg)
    st.session_state.flash_msg = ""

# 全タブ共通のリアルタイム確認パネル
render_live_preview()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. 読み込み・データ管理",
    "2. グラフ可視化 (EDA)",
    "3. 前処理",
    "4. 特徴量作成",
    "5. 機械学習・評価",
    "6. 提出・データ出力 (Submit)"
])

# ==========================================
# TAB 1: データ読み込み・管理・結合
# ==========================================
with tab1:
    t1_mode = st.radio(
        "メニュー",
        ["1. 読み込み・概要", "2. データ管理（追加・削除・保存）", "3. データ結合（横結合・縦結合）"],
        horizontal=True
    )

    # --- 1. 読み込み・概要 ---
    if t1_mode == "1. 読み込み・概要":
        st.subheader("データのアップロード（CSV / Parquet / ZIP対応）")

        if mode == "通常分析モード":
            uploaded_files = st.file_uploader(
                "分析用データをアップロード（複数ファイル同時選択可）",
                type=["csv", "parquet", "zip"],
                accept_multiple_files=True,
                key="normal_multi_uploader"
            )
            if uploaded_files:
                added_names = []
                for f in uploaded_files:
                    if f.name not in st.session_state.loaded_normal_files:
                        try:
                            df_loaded, start_m, end_m = load_file(f)
                            if st.session_state.train_df is None:
                                st.session_state.train_df = df_loaded.copy()
                                added_names.append(f"train ({f.name})")
                            else:
                                base_name = os.path.splitext(f.name)[0]
                                ds_name = base_name if base_name not in ["train", "test"] else f"{base_name}_2"
                                st.session_state.extra_dfs[ds_name] = df_loaded.copy()
                                added_names.append(ds_name)
                            st.session_state.loaded_normal_files.add(f.name)
                        except Exception as e:
                            st.error(f"{f.name} の読み込みエラー: {e}")

                if added_names:
                    notify_and_refresh(f"ファイルを読み込みました: {', '.join(added_names)}")

                if st.button("選択中のファイルを初期状態に再読み込み"):
                    st.session_state.train_df = None
                    st.session_state.extra_dfs = {}
                    st.session_state.loaded_normal_files = set()
                    st.rerun()

        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                train_file = st.file_uploader("学習用データ (train)", type=["csv", "parquet", "zip"])
                if train_file:
                    try:
                        df, start_m, end_m = load_file(train_file)
                        if st.session_state.train_df is None or st.button("trainを初期状態にリセット"):
                            st.session_state.train_df = df.copy()
                            notify_and_refresh(f"Train読み込み完了: {df.shape[0]:,}行 × {df.shape[1]}列 | {start_m:.1f}MB→{end_m:.1f}MB")
                        st.caption(f"読込済み: {df.shape[0]:,}行 × {df.shape[1]}列 | {start_m:.1f}MB→{end_m:.1f}MB")
                    except Exception as e:
                        st.error(f"Train読み込みエラー: {e}")

            with col2:
                test_file = st.file_uploader("評価用データ (test)", type=["csv", "parquet", "zip"])
                if test_file:
                    try:
                        t_df, t_start, t_end = load_file(test_file)
                        if st.session_state.test_df is None or st.button("testを初期状態にリセット"):
                            st.session_state.test_df = t_df.copy()
                            st.session_state.test_raw_df = t_df.copy()
                            notify_and_refresh(f"Test読み込み完了: {t_df.shape[0]:,}行 × {t_df.shape[1]}列 | {t_start:.1f}MB→{t_end:.1f}MB")
                        st.caption(f"読込済み: {t_df.shape[0]:,}行 × {t_df.shape[1]}列 | {t_start:.1f}MB→{t_end:.1f}MB")
                    except Exception as e:
                        st.error(f"Test読み込みエラー: {e}")

            with col3:
                sub_has_header = st.checkbox("Submitにヘッダー（列名）あり", value=True, help="SIGNATEなどヘッダーなしの提出ファイルはオフにしてください")
                sub_file = st.file_uploader("提出用テンプレート (sample_submit)", type=["csv", "parquet", "zip"])
                if sub_file:
                    try:
                        s_df = load_sub_file(sub_file, sub_has_header)
                        st.session_state.sub_template_df = s_df.copy()
                        st.session_state.sub_has_header = sub_has_header
                        st.success(f"Submit読込完了: {s_df.shape[0]:,}行 × {s_df.shape[1]}列")
                    except Exception as e:
                        st.error(f"Submit読み込みエラー: {e}")

        datasets = get_all_datasets()

        # --- 通常・コンペ共通：2データ間の文字列カラム非共通ユニーク値チェック ---
        if len(datasets) >= 2:
            st.divider()
            st.write("### 🔍 2データ間の差異チェック（文字列カラムの非共通ユニーク値）")
            ds_names = list(datasets.keys())
            dc1, dc2 = st.columns(2)
            cmp_a_name = dc1.selectbox("比較元データ (A)", ds_names, index=0, key="cmp_a_name")
            cmp_b_name = dc2.selectbox("比較先データ (B)", ds_names, index=1, key="cmp_b_name")

            df_a = datasets[cmp_a_name]
            df_b = datasets[cmp_b_name]

            str_common_cols = [
                c for c in df_a.columns
                if c in df_b.columns and (
                    pd.api.types.is_object_dtype(df_a[c])
                    or pd.api.types.is_string_dtype(df_a[c])
                    or isinstance(df_a[c].dtype, pd.CategoricalDtype)
                )
            ]

            diff_summary = []
            diff_details = {}
            for c in str_common_cols:
                a_u = set(df_a[c].dropna().astype(str).unique())
                b_u = set(df_b[c].dropna().astype(str).unique())
                a_only = sorted(a_u - b_u)
                b_only = sorted(b_u - a_u)
                if len(a_only) > 0 or len(b_only) > 0:
                    diff_summary.append({
                        "カラム名": c,
                        "共通ユニーク数": len(a_u & b_u),
                        f"{cmp_a_name}のみ(個)": len(a_only),
                        f"{cmp_b_name}のみ(個)": len(b_only),
                        f"{cmp_a_name}のみの値(一部)": ", ".join(a_only[:5]),
                        f"{cmp_b_name}のみの値(一部)": ", ".join(b_only[:5]),
                    })
                    diff_details[c] = (a_only, b_only)

            if diff_summary:
                diff_summary_df = pd.DataFrame(diff_summary)
                st.warning(f"「{cmp_a_name}」と「{cmp_b_name}」の文字列カラムのうち **{len(diff_summary_df)}列** で非共通の値が見つかりました。")
                st.dataframe(diff_summary_df, use_container_width=True)

                sel_diff_col = st.selectbox("非共通の値をすべて見るカラムを選択", list(diff_details.keys()), key="sel_diff_col")
                if sel_diff_col:
                    a_only, b_only = diff_details[sel_diff_col]
                    detail_rows = (
                        [{"区分": f"{cmp_b_name}のみ", "値": v} for v in b_only]
                        + [{"区分": f"{cmp_a_name}のみ", "値": v} for v in a_only]
                    )
                    detail_df = pd.DataFrame(detail_rows)
                    st.caption(f"「{sel_diff_col}」の非共通値: {cmp_b_name}のみ {len(b_only):,}個 / {cmp_a_name}のみ {len(a_only):,}個")
                    st.dataframe(detail_df, use_container_width=True)

                    b1, b2 = st.columns(2)
                    b1.download_button(
                        f"📥 「{sel_diff_col}」の非共通値一覧をCSV出力",
                        data=detail_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"diff_unique_{sel_diff_col}.csv",
                        mime="text/csv",
                        key="dl_diff_detail_csv"
                    )
                    b2.download_button(
                        "📥 全カラムの非共通サマリーをCSV出力",
                        data=diff_summary_df.to_csv(index=False).encode("utf-8"),
                        file_name="diff_unique_summary.csv",
                        mime="text/csv",
                        key="dl_diff_summary_csv"
                    )
            else:
                st.success(f"「{cmp_a_name}」と「{cmp_b_name}」の文字列カラムに非共通ユニーク値はありませんでした。")

        if st.session_state.train_df is not None:
            df_show = st.session_state.train_df
            st.divider()
            st.write("### カラム概要（型・欠損・ユニーク数：Train）")
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

    # --- 2. データ管理（追加・削除・保存） ---
    elif t1_mode == "2. データ管理（追加・削除・保存）":
        st.subheader("➕ 追加データの登録（結合用・サブデータなど）")
        c1, c2 = st.columns([2, 1])
        extra_file = c1.file_uploader("追加で読み込むファイル (CSV / Parquet / ZIP)", type=["csv", "parquet", "zip"], key="extra_uploader")
        default_extra_name = os.path.splitext(extra_file.name)[0] if extra_file else "sub_data"
        extra_name = c2.text_input("登録するデータ名", value=default_extra_name, key="extra_name_input")

        if extra_file and st.button("📥 このデータを登録する"):
            if extra_name in ["train", "test"]:
                st.error("「train」「test」以外の名前を指定してください。")
            else:
                try:
                    e_df, e_start, e_end = load_file(extra_file)
                    st.session_state.extra_dfs[extra_name] = e_df.copy()
                    notify_and_refresh(f"データ「{extra_name}」を登録しました（{e_df.shape[0]:,}行 × {e_df.shape[1]}列）")
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")

        st.divider()
        st.subheader("📂 登録中のデータ一覧・個別保存・個別削除")
        datasets = get_all_datasets()
        if not datasets:
            st.info("まだデータが読み込まれていません。")
        else:
            list_rows = []
            for d_name, d_df in datasets.items():
                list_rows.append({
                    "データ名": d_name,
                    "行数": f"{d_df.shape[0]:,}",
                    "列数": d_df.shape[1],
                    "メモリ(MB)": round(d_df.memory_usage().sum() / 1024**2, 2),
                    "欠損数": f"{d_df.isnull().sum().sum():,}"
                })
            st.dataframe(pd.DataFrame(list_rows), use_container_width=True)

            sel_ds = st.selectbox("保存・削除するデータを選択", list(datasets.keys()), key="manage_sel_ds")
            target_df = datasets[sel_ds]

            st.write(f"#### 「{sel_ds}」の保存・削除")
            mc1, mc2, mc3, mc4 = st.columns([2, 1, 1, 1])
            save_fname = mc1.text_input("保存ファイル名（拡張子不要）", value=sel_ds, key="manage_save_fname")
            mc2.write("")
            mc2.download_button(
                "📥 CSV保存",
                data=target_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{save_fname}.csv",
                mime="text/csv",
                key="manage_dl_csv"
            )
            mc3.write("")
            mc3.download_button(
                "📥 Parquet保存",
                data=target_df.to_parquet(index=False),
                file_name=f"{save_fname}.parquet",
                key="manage_dl_pq"
            )
            mc4.write("")
            if mc4.button("🗑️ このデータを削除", key="manage_del_btn"):
                if sel_ds == "train":
                    st.session_state.train_df = None
                    if os.path.exists(TRAIN_CACHE_PATH):
                        os.remove(TRAIN_CACHE_PATH)
                elif sel_ds == "test":
                    st.session_state.test_df = None
                    st.session_state.test_raw_df = None
                    st.session_state.test_preds = None
                    if os.path.exists(TEST_CACHE_PATH):
                        os.remove(TEST_CACHE_PATH)
                else:
                    st.session_state.extra_dfs.pop(sel_ds, None)
                notify_and_refresh(f"データ「{sel_ds}」を削除しました。")

    # --- 3. データ結合（横結合・縦結合） ---
    elif t1_mode == "3. データ結合（横結合・縦結合）":
        datasets = get_all_datasets()
        if len(datasets) < 2:
            st.info("データ結合を行うには、2つ以上のデータが必要です。「1. 読み込み・概要」または「2. データ管理」から追加データを読み込んでください。")
        else:
            join_mode = st.radio("結合方法", ["横結合 (Merge / Join)", "縦結合 (Concat)"], horizontal=True)

            if join_mode == "横結合 (Merge / Join)":
                ds_names = list(datasets.keys())
                jc1, jc2, jc3 = st.columns(3)
                left_name = jc1.selectbox("左データ（ベース）", ds_names, index=0, key="merge_left")
                right_name = jc2.selectbox("右データ（結合する側）", ds_names, index=min(1, len(ds_names)-1), key="merge_right")
                how_opt = jc3.selectbox("結合タイプ", ["left", "inner", "outer"], key="merge_how")

                l_df = datasets[left_name]
                r_df = datasets[right_name]

                kc1, kc2 = st.columns(2)
                left_key = kc1.selectbox(f"左データ「{left_name}」の結合キー", l_df.columns.tolist(), key="merge_lkey")
                right_key = kc2.selectbox(f"右データ「{right_name}」の結合キー", r_df.columns.tolist(), key="merge_rkey")

                r_cols_avail = [c for c in r_df.columns if c != right_key]
                sel_r_cols = st.multiselect(f"右データ「{right_name}」から結合する列", r_cols_avail, default=r_cols_avail, key="merge_rcols")
                dedup_right = st.checkbox("右データの結合キー重複を除去する（行数が増えるのを防ぐ・推奨）", value=True)

                oc1, oc2 = st.columns(2)
                dest_opt = oc1.radio("結合結果の保存先", [f"左データ「{left_name}」に上書き", "新しいデータとして登録"], horizontal=True, key="merge_dest")
                new_ds_name = oc2.text_input("新しいデータ名（新規登録時）", value=f"{left_name}_merged", key="merge_new_name")

                if st.button("🔗 横結合 (Merge) を実行"):
                    if not sel_r_cols:
                        st.warning("右データから結合する列を1つ以上選んでください。")
                    else:
                        try:
                            l_work = l_df.copy()
                            r_work = r_df[[right_key] + sel_r_cols].copy()
                            l_work["_m_key"] = l_work[left_key].astype(str).str.strip()
                            r_work["_m_key"] = r_work[right_key].astype(str).str.strip()
                            if dedup_right:
                                r_work = r_work.drop_duplicates(subset=["_m_key"])
                            r_work = r_work.drop(columns=[right_key])

                            merged = l_work.merge(r_work, on="_m_key", how=how_opt, suffixes=("", f"_{right_name}")).drop(columns=["_m_key"])

                            if "上書き" in dest_opt:
                                if left_name == "train":
                                    st.session_state.train_df = merged
                                elif left_name == "test":
                                    st.session_state.test_df = merged
                                else:
                                    st.session_state.extra_dfs[left_name] = merged
                                notify_and_refresh(f"「{left_name}」に「{right_name}」を横結合しました！（{merged.shape[0]:,}行 × {merged.shape[1]}列）")
                            else:
                                st.session_state.extra_dfs[new_ds_name] = merged
                                notify_and_refresh(f"結合データ「{new_ds_name}」を新規登録しました！（{merged.shape[0]:,}行 × {merged.shape[1]}列）")
                        except Exception as e:
                            st.error(f"結合エラー: {e}")

            else:
                sel_concat = st.multiselect("縦に結合するデータを2つ以上選択", list(datasets.keys()), default=list(datasets.keys())[:2], key="concat_sel")
                cc1, cc2 = st.columns(2)
                concat_dest = cc1.radio("結合結果の保存先", ["新しいデータとして登録", "Trainデータに上書き"], horizontal=True, key="concat_dest")
                concat_new_name = cc2.text_input("新しいデータ名（新規登録時）", value="concat_data", key="concat_new_name")

                if st.button("🔗 縦結合 (Concat) を実行"):
                    if len(sel_concat) < 2:
                        st.warning("2つ以上のデータを選択してください。")
                    else:
                        try:
                            concated = pd.concat([datasets[n] for n in sel_concat], ignore_index=True)
                            if concat_dest == "Trainデータに上書き":
                                st.session_state.train_df = concated
                                notify_and_refresh(f"縦結合したデータでTrainを上書きしました！（{concated.shape[0]:,}行 × {concated.shape[1]}列）")
                            else:
                                st.session_state.extra_dfs[concat_new_name] = concated
                                notify_and_refresh(f"縦結合データ「{concat_new_name}」を新規登録しました！（{concated.shape[0]:,}行 × {concated.shape[1]}列）")
                        except Exception as e:
                            st.error(f"結合エラー: {e}")

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
# TAB 3: 前処理（基本・日付分解・文字列抽出・条件抽出・特定行削除）
# ==========================================
with tab3:
    if st.session_state.train_df is None:
        st.info("先にタブ1でデータを読み込んでください。")
    else:
        df = st.session_state.train_df
        all_cols = df.columns.tolist()

        prep_mode = st.radio(
            "前処理メニュー",
            [
                "1. 列削除・欠損補完",
                "2. 日付・時刻の分解",
                "3. 文字列の抜き取り・加工",
                "4. 条件で抽出・出力",
                "5. 特定行の削除"
            ],
            horizontal=True
        )

        if prep_mode == "1. 列削除・欠損補完":
            st.write("### 欠損値の確認")
            null_df = pd.DataFrame({"欠損数": df.isnull().sum(), "欠損率(%)": (df.isnull().mean() * 100).round(2), "型": df.dtypes.astype(str)})
            st.dataframe(null_df[null_df["欠損数"] > 0], use_container_width=True)

            st.write("### 列削除・欠損補完の実行")
            drop_cols = st.multiselect("削除する列（ID列など）", all_cols)
            fill_method = st.selectbox("数値列の欠損補完", ["何もしない（LightGBMに任せる）", "中央値で補完", "平均値で補完", "0で補完"])

            if st.button("基本前処理を適用する"):
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
                notify_and_refresh("基本前処理を適用しました！（リアルタイム確認も更新済み）")

        elif prep_mode == "2. 日付・時刻の分解":
            st.write("### 日付列を年・月・日・曜日などに分解")
            dt_col = st.selectbox("日付・時刻の列を選択", all_cols, key="dt_col")
            dt_parts = st.multiselect(
                "作成する項目を選択",
                ["年 (year)", "月 (month)", "日 (day)", "曜日 (dayofweek: 0=月〜6=日)", "土日フラグ (is_weekend)", "時 (hour)"],
                default=["年 (year)", "月 (month)", "日 (day)", "曜日 (dayofweek: 0=月〜6=日)"]
            )
            drop_orig_dt = st.checkbox("分解後、元の日付列を削除する", value=False)

            if st.button("日付を分解する"):
                if not dt_parts:
                    st.warning("作成する項目を1つ以上選んでください。")
                else:
                    def apply_dt(target_df):
                        if dt_col not in target_df.columns:
                            return target_df
                        s = pd.to_datetime(target_df[dt_col], errors="coerce")
                        if "年 (year)" in dt_parts:
                            target_df[f"{dt_col}_year"] = s.dt.year.astype("float32")
                        if "月 (month)" in dt_parts:
                            target_df[f"{dt_col}_month"] = s.dt.month.astype("float32")
                        if "日 (day)" in dt_parts:
                            target_df[f"{dt_col}_day"] = s.dt.day.astype("float32")
                        if "曜日 (dayofweek: 0=月〜6=日)" in dt_parts:
                            target_df[f"{dt_col}_dayofweek"] = s.dt.dayofweek.astype("float32")
                        if "土日フラグ (is_weekend)" in dt_parts:
                            target_df[f"{dt_col}_is_weekend"] = (s.dt.dayofweek >= 5).astype("int8")
                        if "時 (hour)" in dt_parts:
                            target_df[f"{dt_col}_hour"] = s.dt.hour.astype("float32")
                        if drop_orig_dt:
                            target_df = target_df.drop(columns=[dt_col])
                        return target_df

                    st.session_state.train_df = apply_dt(st.session_state.train_df)
                    if st.session_state.test_df is not None:
                        st.session_state.test_df = apply_dt(st.session_state.test_df)
                    notify_and_refresh(f"「{dt_col}」を分解しました！（リアルタイム確認も更新済み）")

        elif prep_mode == "3. 文字列の抜き取り・加工":
            st.write("### 特定の文字列を抜き取る・フラグ化する")
            str_col = st.selectbox("対象の列を選択", all_cols, key="str_col")
            str_op = st.selectbox(
                "抜き取り方法",
                [
                    "数値だけを抜き取る（例: 徒歩10分 → 10）",
                    "特定の文字を含むかフラグ化 (0/1)",
                    "区切り文字で分割してN番目を取得",
                    "先頭または末尾からN文字を抜き取る",
                    "正規表現で抜き取る（上級者向け）"
                ]
            )

            param1, param2 = None, None
            default_new_name = f"{str_col}_extracted"

            if "数値だけ" in str_op:
                default_new_name = f"{str_col}_num"
            elif "含むかフラグ" in str_op:
                param1 = st.text_input("検索する文字列（例: 駅、東京 など）", value="")
                default_new_name = f"{str_col}_has_{param1}" if param1 else f"{str_col}_flag"
            elif "区切り文字" in str_op:
                c1, c2 = st.columns(2)
                param1 = c1.text_input("区切り文字（例: / , - スペース など）", value="-")
                param2 = c2.number_input("何番目を取得するか（0が先頭）", min_value=0, max_value=20, value=0, step=1)
                default_new_name = f"{str_col}_part{int(param2)}"
            elif "先頭または末尾" in str_op:
                c1, c2 = st.columns(2)
                param1 = c1.radio("位置", ["先頭から", "末尾から"], horizontal=True)
                param2 = c2.number_input("文字数", min_value=1, max_value=50, value=3, step=1)
                default_new_name = f"{str_col}_{'head' if param1 == '先頭から' else 'tail'}{int(param2)}"
            elif "正規表現" in str_op:
                param1 = st.text_input("正規表現パターン（抜き出したい部分を () で囲む 例: ([A-Z]+)）", value=r"(\d+)")
                default_new_name = f"{str_col}_regex"

            new_str_col = st.text_input("新しく作る列名", value=default_new_name)

            if st.button("文字列の抜き取りを実行"):
                try:
                    def apply_str(target_df):
                        if str_col not in target_df.columns:
                            return target_df
                        s = target_df[str_col].astype(str)
                        if "数値だけ" in str_op:
                            ext = s.str.extract(r"(-?\d+\.?\d*)")[0]
                            target_df[new_str_col] = pd.to_numeric(ext, errors="coerce").astype("float32")
                        elif "含むかフラグ" in str_op:
                            target_df[new_str_col] = s.str.contains(param1, na=False, regex=False).astype("int8")
                        elif "区切り文字" in str_op:
                            target_df[new_str_col] = s.str.split(param1).str.get(int(param2)).astype("category")
                        elif "先頭または末尾" in str_op:
                            n = int(param2)
                            if param1 == "先頭から":
                                target_df[new_str_col] = s.str[:n].astype("category")
                            else:
                                target_df[new_str_col] = s.str[-n:].astype("category")
                        elif "正規表現" in str_op:
                            ext = s.str.extract(param1)[0]
                            target_df[new_str_col] = ext.astype("category")
                        return target_df

                    st.session_state.train_df = apply_str(st.session_state.train_df)
                    if st.session_state.test_df is not None:
                        st.session_state.test_df = apply_str(st.session_state.test_df)
                    notify_and_refresh(f"新しい列「{new_str_col}」を作成しました！（リアルタイム確認も更新済み）")
                except Exception as e:
                    st.error(f"処理中にエラーが発生しました: {e}")

        elif prep_mode == "4. 条件で抽出・出力":
            st.write("### 条件を指定して該当する行だけを抽出・出力")
            target_options = ["Trainデータ"] + (["Testデータ"] if st.session_state.test_df is not None else [])
            fc1, fc2, fc3 = st.columns(3)
            flt_target = fc1.selectbox("対象データ", target_options, key="flt_target")
            n_conds = fc2.selectbox("条件の数", [1, 2, 3], index=0, key="flt_n_conds")
            logic_op = fc3.radio("複数条件のつなぎ方", ["AND（すべて満たす）", "OR（いずれかを満たす）"], horizontal=True, key="flt_logic")

            src_df = st.session_state.train_df if flt_target == "Trainデータ" else st.session_state.test_df
            src_cols = src_df.columns.tolist()

            masks = []
            for i in range(n_conds):
                st.markdown(f"**条件 {i + 1}**")
                cc1, cc2, cc3 = st.columns(3)
                f_col = cc1.selectbox("カラム", src_cols, key=f"flt_col_{i}")
                is_num = pd.api.types.is_numeric_dtype(src_df[f_col])

                if is_num:
                    ops = ["以上 (>=)", "より大きい (>)", "以下 (<=)", "未満 (<)", "等しい (==)", "等しくない (!=)", "欠損値である (NaN)", "欠損値でない"]
                else:
                    ops = ["を含む", "を含まない", "完全一致 (==)", "一致しない (!=)", "欠損値である (NaN)", "欠損値でない"]

                f_op = cc2.selectbox("条件", ops, key=f"flt_op_{i}")

                if "欠損値" in f_op:
                    cc3.write("")
                    m = src_df[f_col].isnull() if "である" in f_op else src_df[f_col].notnull()
                elif is_num:
                    f_val = cc3.number_input("値（数値）", value=0.0, format="%g", key=f"flt_num_{i}")
                    if f_op == "以上 (>=)":
                        m = src_df[f_col] >= f_val
                    elif f_op == "より大きい (>)":
                        m = src_df[f_col] > f_val
                    elif f_op == "以下 (<=)":
                        m = src_df[f_col] <= f_val
                    elif f_op == "未満 (<)":
                        m = src_df[f_col] < f_val
                    elif f_op == "等しい (==)":
                        m = src_df[f_col] == f_val
                    else:
                        m = src_df[f_col] != f_val
                else:
                    f_val = cc3.text_input("値（文字列）", value="", key=f"flt_str_{i}")
                    s_str = src_df[f_col].astype(str)
                    if f_op == "を含む":
                        m = s_str.str.contains(f_val, na=False, regex=False)
                    elif f_op == "を含まない":
                        m = ~s_str.str.contains(f_val, na=False, regex=False)
                    elif f_op == "完全一致 (==)":
                        m = s_str == f_val
                    else:
                        m = s_str != f_val

                masks.append(m.fillna(False))

            final_mask = masks[0]
            for m in masks[1:]:
                final_mask = (final_mask & m) if "AND" in logic_op else (final_mask | m)

            filtered_df = src_df[final_mask].copy()
            ratio = (len(filtered_df) / len(src_df) * 100) if len(src_df) > 0 else 0.0
            st.success(f"抽出結果: **{len(filtered_df):,}行** / {len(src_df):,}行（{ratio:.2f}%）")
            st.dataframe(filtered_df.head(50), use_container_width=True)

            st.write("#### 抽出したデータの保存・反映")
            out_name = st.text_input("抽出データのファイル名（拡張子不要）", value="filtered_data", key="flt_out_name")
            fb1, fb2, fb3 = st.columns(3)
            fb1.download_button(
                "📥 抽出データをCSVで保存",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{out_name}.csv",
                mime="text/csv",
                key="flt_dl_csv"
            )
            fb2.download_button(
                "📥 抽出データをParquetで保存",
                data=filtered_df.to_parquet(index=False),
                file_name=f"{out_name}.parquet",
                key="flt_dl_pq"
            )
            if fb3.button(f"⚠️ この抽出結果で{flt_target}を上書き", key="flt_overwrite"):
                if flt_target == "Trainデータ":
                    st.session_state.train_df = filtered_df.reset_index(drop=True)
                else:
                    st.session_state.test_df = filtered_df.reset_index(drop=True)
                notify_and_refresh(f"{flt_target}を抽出結果（{len(filtered_df):,}行）で上書きしました！")

        elif prep_mode == "5. 特定行の削除":
            st.write("### 特定の行を削除する")
            del_target_opts = ["Trainデータ"] + (["Testデータ（注意：提出行数が変わります）"] if st.session_state.test_df is not None else [])
            dc1, dc2 = st.columns(2)
            del_target = dc1.selectbox("対象データ", del_target_opts, key="del_row_target")
            del_method = dc2.selectbox(
                "削除方法",
                [
                    "A. 行番号（インデックス）を指定して削除",
                    "B. 条件に一致する行を削除（外れ値・ノイズ除去）",
                    "C. 重複行を削除",
                    "D. 指定した列が欠損 (NaN) の行を削除"
                ]
            )

            is_train_del = del_target.startswith("Train")
            target_df = st.session_state.train_df if is_train_del else st.session_state.test_df
            t_cols = target_df.columns.tolist()

            if del_method.startswith("A."):
                idx_text = st.text_input("削除したい行番号を入力（カンマ区切り・ハイフン範囲指定可 例: 0, 3, 10-15）", value="")
                del_indices = parse_row_indices(idx_text, len(target_df))
                if del_indices:
                    st.caption(f"削除対象: **{len(del_indices):,}行**（プレビュー先頭10件）")
                    st.dataframe(target_df.iloc[del_indices[:10]], use_container_width=True)
                if st.button("🗑️ 指定した行番号を削除する"):
                    if not del_indices:
                        st.warning("有効な行番号を入力してください。")
                    else:
                        new_df = target_df.drop(index=target_df.index[del_indices]).reset_index(drop=True)
                        if is_train_del:
                            st.session_state.train_df = new_df
                        else:
                            st.session_state.test_df = new_df
                        notify_and_refresh(f"{len(del_indices):,}行を削除しました（残り {len(new_df):,}行）")

            elif del_method.startswith("B."):
                cc1, cc2, cc3 = st.columns(3)
                d_col = cc1.selectbox("カラム", t_cols, key="del_cond_col")
                is_num = pd.api.types.is_numeric_dtype(target_df[d_col])
                ops = ["以上 (>=)", "より大きい (>)", "以下 (<=)", "未満 (<)", "等しい (==)"] if is_num else ["を含む", "完全一致 (==)"]
                d_op = cc2.selectbox("削除する条件", ops, key="del_cond_op")

                if is_num:
                    d_val = cc3.number_input("値（数値）", value=0.0, format="%g", key="del_cond_num")
                    if d_op == "以上 (>=)":
                        del_mask = target_df[d_col] >= d_val
                    elif d_op == "より大きい (>)":
                        del_mask = target_df[d_col] > d_val
                    elif d_op == "以下 (<=)":
                        del_mask = target_df[d_col] <= d_val
                    elif d_op == "未満 (<)":
                        del_mask = target_df[d_col] < d_val
                    else:
                        del_mask = target_df[d_col] == d_val
                else:
                    d_val = cc3.text_input("値（文字列）", value="", key="del_cond_str")
                    s_str = target_df[d_col].astype(str)
                    del_mask = s_str.str.contains(d_val, na=False, regex=False) if d_op == "を含む" else (s_str == d_val)

                del_mask = del_mask.fillna(False)
                del_cnt = int(del_mask.sum())
                st.caption(f"削除対象の行数: **{del_cnt:,}行** / {len(target_df):,}行")
                if st.button("🗑️ 条件に一致する行を削除する"):
                    if del_cnt == 0:
                        st.warning("削除対象の行がありません。")
                    else:
                        new_df = target_df[~del_mask].reset_index(drop=True)
                        if is_train_del:
                            st.session_state.train_df = new_df
                        else:
                            st.session_state.test_df = new_df
                        notify_and_refresh(f"条件一致の {del_cnt:,}行を削除しました（残り {len(new_df):,}行）")

            elif del_method.startswith("C."):
                dup_cols = st.multiselect("重複判定に使う列（空欄なら全列一致で判定）", t_cols, key="del_dup_cols")
                subset_arg = dup_cols if dup_cols else None
                dup_cnt = int(target_df.duplicated(subset=subset_arg).sum())
                st.caption(f"重複している行数: **{dup_cnt:,}行**")
                if st.button("🗑️ 重複行を削除する"):
                    if dup_cnt == 0:
                        st.warning("重複行はありません。")
                    else:
                        new_df = target_df.drop_duplicates(subset=subset_arg).reset_index(drop=True)
                        if is_train_del:
                            st.session_state.train_df = new_df
                        else:
                            st.session_state.test_df = new_df
                        notify_and_refresh(f"重複行 {dup_cnt:,}行を削除しました（残り {len(new_df):,}行）")

            elif del_method.startswith("D."):
                na_cols = st.multiselect("欠損チェック対象の列（空欄ならいずれかの列が欠損の行）", t_cols, key="del_na_cols")
                subset_arg = na_cols if na_cols else None
                na_cnt = int(target_df.isnull().any(axis=1).sum()) if subset_arg is None else int(target_df[subset_arg].isnull().any(axis=1).sum())
                st.caption(f"削除対象の欠損行数: **{na_cnt:,}行**")
                if st.button("🗑️ 欠損を含む行を削除する"):
                    if na_cnt == 0:
                        st.warning("対象の欠損行はありません。")
                    else:
                        new_df = target_df.dropna(subset=subset_arg).reset_index(drop=True)
                        if is_train_del:
                            st.session_state.train_df = new_df
                        else:
                            st.session_state.test_df = new_df
                        notify_and_refresh(f"欠損行 {na_cnt:,}行を削除しました（残り {len(new_df):,}行）")

        render_custom_save_ui("tab3")

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
            [
                "1. 四則演算（比率・差分など）",
                "2. GroupBy集約（カテゴリ別集計）",
                "3. 時系列（ラグ・差分・移動平均）",
                "4. 地域メッシュ（国土交通）"
            ],
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
                    notify_and_refresh(f"特徴量「{new_col_name}」を追加しました！")

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
                        notify_and_refresh(f"GroupBy特徴量（{len(agg_funcs)}列）を追加しました！")

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

                    notify_and_refresh(f"時系列特徴量を {len(created_cols)} 列追加しました！")

        elif fe_type == "4. 地域メッシュ（国土交通）":
            mesh_mode = st.selectbox(
                "メッシュ処理を選択",
                [
                    "A. メッシュコード → 緯度経度・上位メッシュ分解（8〜10桁対応）",
                    "B. 緯度・経度 → 地域メッシュコード生成（1km / 500m）",
                    "C. 外部メッシュデータ（国土数値情報等）を結合"
                ]
            )

            if mesh_mode.startswith("A."):
                mesh_col = st.selectbox("メッシュコードの列", all_cols, key="mesh_dec_col")
                mesh_opts = st.multiselect(
                    "作成する項目",
                    ["中心緯度・経度 (lat, lon)", "1次メッシュ (4桁・約80km)", "2次メッシュ (6桁・約10km)", "3次メッシュ (8桁・約1km)"],
                    default=["中心緯度・経度 (lat, lon)", "2次メッシュ (6桁・約10km)", "3次メッシュ (8桁・約1km)"]
                )
                if st.button("メッシュコードを分解・変換する"):
                    def apply_mesh_dec(target_df):
                        if mesh_col not in target_df.columns:
                            return target_df
                        s_str = target_df[mesh_col].astype(str).str.split(".").str[0].str.strip()
                        if "中心緯度・経度 (lat, lon)" in mesh_opts:
                            coords = s_str.map(decode_mesh_code)
                            target_df[f"{mesh_col}_lat"] = [c[0] for c in coords]
                            target_df[f"{mesh_col}_lon"] = [c[1] for c in coords]
                            target_df[f"{mesh_col}_lat"] = target_df[f"{mesh_col}_lat"].astype("float32")
                            target_df[f"{mesh_col}_lon"] = target_df[f"{mesh_col}_lon"].astype("float32")
                        if "1次メッシュ (4桁・約80km)" in mesh_opts:
                            target_df[f"{mesh_col}_m1"] = s_str.str[:4].astype("category")
                        if "2次メッシュ (6桁・約10km)" in mesh_opts:
                            target_df[f"{mesh_col}_m2"] = s_str.str[:6].astype("category")
                        if "3次メッシュ (8桁・約1km)" in mesh_opts:
                            target_df[f"{mesh_col}_m3"] = s_str.str[:8].astype("category")
                        return target_df

                    st.session_state.train_df = apply_mesh_dec(st.session_state.train_df)
                    if st.session_state.test_df is not None:
                        st.session_state.test_df = apply_mesh_dec(st.session_state.test_df)
                    notify_and_refresh(f"「{mesh_col}」のメッシュ分解・変換を完了しました！")

            elif mesh_mode.startswith("B."):
                if len(num_cols) < 2:
                    st.warning("緯度・経度の数値列が2つ以上必要です。")
                else:
                    c1, c2, c3 = st.columns(3)
                    lat_col = c1.selectbox("緯度 (lat) の列", num_cols, key="m_lat")
                    lon_col = c2.selectbox("経度 (lon) の列", num_cols, index=min(1, len(num_cols)-1), key="m_lon")
                    m_level = c3.selectbox("作成するメッシュ精度", ["4次メッシュ (9桁・500m)", "3次メッシュ (8桁・1km)"])

                    if st.button("緯度経度からメッシュコードを生成"):
                        lvl = 4 if "4次" in m_level else 3
                        new_m_col = f"mesh_{'500m' if lvl == 4 else '1km'}"

                        def apply_mesh_enc(target_df):
                            if lat_col in target_df.columns and lon_col in target_df.columns:
                                target_df[new_m_col] = encode_latlon_to_mesh(target_df[lat_col], target_df[lon_col], level=lvl).astype("category")
                            return target_df

                        st.session_state.train_df = apply_mesh_enc(st.session_state.train_df)
                        if st.session_state.test_df is not None:
                            st.session_state.test_df = apply_mesh_enc(st.session_state.test_df)
                        notify_and_refresh(f"メッシュコード列「{new_m_col}」を生成しました！")

            elif mesh_mode.startswith("C."):
                ext_file = st.file_uploader("外部メッシュデータ (CSV / Parquet / ZIP)", type=["csv", "parquet", "zip"], key="ext_mesh_file")
                if ext_file:
                    ext_df, _, _ = load_file(ext_file)
                    st.caption(f"外部データ: {ext_df.shape[0]:,}行 × {ext_df.shape[1]}列")
                    c1, c2 = st.columns(2)
                    left_key = c1.selectbox("Train/Test側の結合キー（メッシュ列など）", all_cols, key="ext_lkey")
                    right_key = c2.selectbox("外部データ側の結合キー", ext_df.columns.tolist(), key="ext_rkey")
                    add_cols = st.multiselect(
                        "結合して追加する外部データの列",
                        [c for c in ext_df.columns if c != right_key],
                        default=[c for c in ext_df.columns if c != right_key][:5]
                    )

                    if st.button("外部メッシュデータを結合 (Left Join)"):
                        if not add_cols:
                            st.warning("追加する列を1つ以上選んでください。")
                        else:
                            ext_sub = ext_df[[right_key] + add_cols].copy()
                            ext_sub["_join_key"] = ext_sub[right_key].astype(str).str.split(".").str[0].str.strip()
                            ext_sub = ext_sub.drop_duplicates(subset=["_join_key"]).drop(columns=[right_key])

                            def apply_ext_join(target_df):
                                if left_key not in target_df.columns:
                                    return target_df
                                target_df = target_df.copy()
                                target_df["_join_key"] = target_df[left_key].astype(str).str.split(".").str[0].str.strip()
                                merged = target_df.merge(ext_sub, on="_join_key", how="left", suffixes=("", "_ext"))
                                return merged.drop(columns=["_join_key"])

                            st.session_state.train_df = apply_ext_join(st.session_state.train_df)
                            if st.session_state.test_df is not None:
                                st.session_state.test_df = apply_ext_join(st.session_state.test_df)
                            notify_and_refresh(f"外部データから {len(add_cols)} 列を結合しました！")

        render_custom_save_ui("tab4")

# ==========================================
# TAB 5: 機械学習・評価（分割方法指定・チューニング対応）
# ==========================================
with tab5:
    if st.session_state.train_df is None:
        st.info("先にタブ1でデータを読み込んでください。")
    else:
        df = st.session_state.train_df
        all_cols = df.columns.tolist()

        c1, c2, c3 = st.columns(3)
        target_col = c1.selectbox("目的変数 (Target)", all_cols, index=len(all_cols)-1)
        task_type = c2.selectbox("タスク種別", ["回帰", "二値分類"])
        model_name = c3.selectbox(
            "使用モデル",
            ["LightGBM", "決定木 (DecisionTree)", "ランダムフォレスト (RandomForest)", "線形モデル (Ridge / Logistic)"]
        )

        tune_preset = st.selectbox(
            "チューニング設定（浅い・深いなど）",
            [
                "標準（バランス）",
                "浅い（過学習防止）",
                "深い（表現力重視）",
                "高精度・じっくり（コンペ向け）",
                "🔥 全パターン一括比較（浅い・標準・深い）"
            ]
        )

        default_features = [c for c in all_cols if c != target_col]
        feature_cols = st.multiselect("学習に使う特徴量カラムを選択", default_features, default=default_features)

        c4, c5, c6 = st.columns(3)
        cv_type = c4.selectbox(
            "データ分割方法",
            [
                "KFold（ランダム分割）",
                "StratifiedKFold（層化抽出・分類向け）",
                "GroupKFold（グループ単位で分割）",
                "TimeSeriesSplit（時系列分割）",
                "Hold-out（単一分割・高速テスト）"
            ]
        )

        if "Hold-out" in cv_type:
            val_ratio = c5.slider("検証データの割合", 0.1, 0.5, 0.2, 0.05)
            n_splits = 1
        else:
            n_splits = c5.slider("CV分割数", 2, 10, 5)
            val_ratio = 0.2

        if task_type == "回帰":
            use_log1p = c6.checkbox("目的変数を log1p 変換（RMSLE最適化）", value=False)
        else:
            use_log1p = False
            c6.write("")

        ts_cv_sort_col = None
        group_cv_col = None
        if "TimeSeriesSplit" in cv_type:
            ts_cv_sort_col = st.selectbox("時系列分割用の並び替え列（日付・時刻など）", ["並び替えない（現在の行順）"] + all_cols)
        elif "GroupKFold" in cv_type:
            group_cv_col = st.selectbox("グループ分割のキー列（ID・メッシュ・店舗など）", all_cols)

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

                if task_type == "二値分類":
                    if y.nunique() != 2:
                        st.error(f"「二値分類」が選ばれていますが、目的変数 '{target_col}' のユニーク数が {y.nunique()} 個です。")
                        st.stop()
                    if not pd.api.types.is_numeric_dtype(y):
                        y = pd.Series(pd.factorize(y)[0])

                if use_log1p and (y < 0).any():
                    st.error("目的変数に負の値が含まれているため、log1p変換（RMSLE最適化）は使用できません。")
                    st.stop()

                if mode == "コンペ用モード" and st.session_state.test_df is not None:
                    missing_in_test = [c for c in feature_cols if c not in st.session_state.test_df.columns]
                    if missing_in_test:
                        st.error(f"testデータに以下の選択カラムが存在しません: {missing_in_test}")
                        st.stop()
                    X_test = st.session_state.test_df[feature_cols].copy()
                else:
                    X_test = None

                if model_name == "LightGBM":
                    for col in X.select_dtypes(include=["object", "string"]).columns:
                        X[col] = X[col].astype("category")
                        if X_test is not None:
                            X_test[col] = X_test[col].astype("category")
                else:
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

                if "Hold-out" in cv_type:
                    strat_arg = y if task_type == "二値分類" else None
                    tr_idx, va_idx = train_test_split(np.arange(len(work_df)), test_size=val_ratio, random_state=42, stratify=strat_arg)
                    splits = [(tr_idx, va_idx)]
                elif "TimeSeriesSplit" in cv_type:
                    splitter = TimeSeriesSplit(n_splits=n_splits)
                    splits = list(splitter.split(X))
                elif "GroupKFold" in cv_type:
                    groups = work_df[group_cv_col].astype(str).fillna("NaN")
                    if groups.nunique() < n_splits:
                        st.error(f"グループ列 '{group_cv_col}' のユニーク数（{groups.nunique()}）がCV分割数（{n_splits}）より少ないため分割できません。")
                        st.stop()
                    splitter = GroupKFold(n_splits=n_splits)
                    splits = list(splitter.split(X, y, groups=groups))
                elif "StratifiedKFold" in cv_type:
                    if task_type == "回帰":
                        y_bins = pd.qcut(y, q=min(10, y.nunique()), labels=False, duplicates="drop")
                        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        splits = list(splitter.split(X, y_bins))
                    else:
                        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        splits = list(splitter.split(X, y))
                else:
                    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    splits = list(splitter.split(X))

                presets_to_run = (
                    ["浅い（過学習防止）", "標準（バランス）", "深い（表現力重視）"]
                    if "一括比較" in tune_preset
                    else [tune_preset]
                )

                results = []
                total_steps = len(presets_to_run) * len(splits)
                step_cnt = 0
                progress = st.progress(0)

                for preset in presets_to_run:
                    oof_preds = np.full(len(work_df), np.nan)
                    test_preds = np.zeros(len(X_test)) if X_test is not None else None
                    importances = np.zeros(len(feature_cols))

                    for fold, (tr_idx, va_idx) in enumerate(splits):
                        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

                        y_tr_fit = np.log1p(y_tr) if use_log1p else y_tr
                        y_va_fit = np.log1p(y_va) if use_log1p else y_va

                        model = build_model(model_name, task_type, preset, 42 + fold)

                        if model_name == "LightGBM":
                            model.fit(X_tr, y_tr_fit, eval_set=[(X_va, y_va_fit)], callbacks=[lgb.early_stopping(50, verbose=False)])
                        else:
                            model.fit(X_tr, y_tr_fit)

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

                        if hasattr(model, "feature_importances_"):
                            importances += model.feature_importances_ / len(splits)
                        elif hasattr(model, "named_steps"):
                            est = list(model.named_steps.values())[-1]
                            coef = np.abs(est.coef_).ravel()
                            importances += coef / len(splits)

                        step_cnt += 1
                        progress.progress(step_cnt / total_steps)

                    valid_mask = ~np.isnan(oof_preds)
                    y_eval = y[valid_mask]
                    p_eval = oof_preds[valid_mask]

                    if task_type == "回帰":
                        rmse = root_mean_squared_error(y_eval, p_eval)
                        mae = mean_absolute_error(y_eval, p_eval)
                        r2 = r2_score(y_eval, p_eval)
                        rmsle = root_mean_squared_log_error(y_eval, np.clip(p_eval, 0, None)) if (y_eval >= 0).all() else np.nan
                        main_score = rmsle if (use_log1p and not np.isnan(rmsle)) else rmse
                        results.append({
                            "設定": preset, "RMSE": rmse, "MAE": mae, "R2": r2, "RMSLE": rmsle,
                            "_key": main_score, "_test_preds": test_preds, "_imp": importances
                        })
                    else:
                        auc = roc_auc_score(y_eval, p_eval)
                        pred_cls = (p_eval >= 0.5).astype(int)
                        acc = accuracy_score(y_eval, pred_cls)
                        f1 = f1_score(y_eval, pred_cls, zero_division=0)
                        ll = log_loss(y_eval, np.clip(p_eval, 1e-15, 1 - 1e-15))
                        results.append({
                            "設定": preset, "ROC-AUC": auc, "Accuracy": acc, "F1": f1, "LogLoss": ll,
                            "_key": -auc, "_test_preds": test_preds, "_imp": importances
                        })

                best_res = min(results, key=lambda x: x["_key"])

                if len(results) > 1:
                    st.write("### 🏆 チューニングパターン一括比較")
                    comp_cols = ["設定", "RMSE", "MAE", "R2", "RMSLE"] if task_type == "回帰" else ["設定", "ROC-AUC", "Accuracy", "F1", "LogLoss"]
                    comp_df = pd.DataFrame([{k: r[k] for k in comp_cols} for r in results])
                    st.dataframe(comp_df, use_container_width=True)
                    st.success(f"最良パターン: **{best_res['設定']}** の予測結果を採用しました！")

                if best_res["_test_preds"] is not None:
                    st.session_state.test_preds = best_res["_test_preds"]
                    st.session_state.last_model_name = f"{model_name.split()[0]}_{best_res['設定'].split('（')[0]}"
                    st.session_state.last_task_type = task_type

                st.write(f"### 📈 評価スコア (OOF: {best_res['設定']} / {cv_type.split('（')[0]})")
                if task_type == "回帰":
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("RMSE（低いほど良い）", f"{best_res['RMSE']:.5f}")
                    m2.metric("MAE（低いほど良い）", f"{best_res['MAE']:.5f}")
                    m3.metric("R²（1に近いほど良い）", f"{best_res['R2']:.5f}")
                    m4.metric("RMSLE（低いほど良い）", f"{best_res['RMSLE']:.5f}" if not np.isnan(best_res["RMSLE"]) else "算出不可")
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("ROC-AUC（高いほど良い）", f"{best_res['ROC-AUC']:.5f}")
                    m2.metric("Accuracy", f"{best_res['Accuracy']:.5f}")
                    m3.metric("F1 Score", f"{best_res['F1']:.5f}")
                    m4.metric("LogLoss（低いほど良い）", f"{best_res['LogLoss']:.5f}")

                imp_title = f"特徴量重要度 Top 20 ({best_res['設定']})" if model_name != "線形モデル (Ridge / Logistic)" else f"回帰係数の絶対値 Top 20 ({best_res['設定']})"
                imp_df = pd.DataFrame({"特徴量": feature_cols, "重要度": best_res["_imp"]}).sort_values("重要度", ascending=True).tail(20)
                fig_imp = px.bar(imp_df, x="重要度", y="特徴量", orientation="h", title=imp_title)
                st.plotly_chart(fig_imp, use_container_width=True)

                if best_res["_test_preds"] is not None:
                    st.info("予測結果を保存しました！「6. 提出・データ出力 (Submit)」タブから提出ファイルをダウンロードできます。")

            except Exception as e:
                st.error(f"学習中にエラーが発生しました: {e}")

# ==========================================
# TAB 6: 提出・データ出力 (Submit)
# ==========================================
with tab6:
    st.subheader("📦 加工済みデータの出力 (Train / Test)")
    if st.session_state.train_df is None:
        st.info("先にタブ1でデータを読み込んでください。")
    else:
        render_custom_save_ui("tab6")

    st.divider()
    st.subheader("🏆 コンペ提出ファイル (Submit) の作成・出力")

    if st.session_state.test_preds is None:
        st.info("コンペ用モードでTestデータを読み込み、タブ5で学習を実行すると、ここで提出ファイルを作成できます。")
    else:
        preds = st.session_state.test_preds.copy()

        st.write("#### 1. 予測値の後処理（任意）")
        p1, p2, p3 = st.columns(3)
        clip_zero = p1.checkbox("0未満を0にクリップ", value=False)
        round_int = p2.checkbox("四捨五入して整数化", value=False)
        binarize = p3.checkbox("しきい値で 0/1 に変換（分類用）", value=False)

        if clip_zero:
            preds = np.clip(preds, 0, None)
        if binarize:
            th = st.slider("0/1 変換のしきい値", 0.0, 1.0, 0.5, 0.01)
            preds = (preds >= th).astype(int)
        elif round_int:
            preds = np.rint(preds).astype(int)

        st.write("#### 2. フォーマット設定")
        if st.session_state.sub_template_df is not None:
            sub_df = st.session_state.sub_template_df.copy()
            if len(sub_df) != len(preds):
                st.error(f"行数が一致しません！ sample_submit: {len(sub_df):,}行 / Test予測: {len(preds):,}行")
                st.stop()

            c1, c2 = st.columns(2)
            target_sub_col = c1.selectbox(
                "予測値を書き込む列（通常は一番右の列）",
                sub_df.columns.tolist(),
                index=len(sub_df.columns) - 1
            )
            include_header = c2.checkbox(
                "ヘッダー（列名）を出力する",
                value=st.session_state.sub_has_header
            )
            sub_df[target_sub_col] = preds
            st.success("読み込んだ sample_submit のフォーマットに予測値を埋め込みました。")
        else:
            st.caption("※タブ1で sample_submit を読み込んでいないため、Testデータから作成します。")
            raw_test = st.session_state.test_raw_df
            c1, c2, c3 = st.columns(3)
            id_col_options = ["なし（予測値のみ）"] + (raw_test.columns.tolist() if raw_test is not None else [])
            id_col = c1.selectbox("ID列を選択（元Testデータから取得）", id_col_options, index=1 if len(id_col_options) > 1 else 0)
            pred_col_name = c2.text_input("予測列の列名", value="prediction")
            include_header = c3.checkbox("ヘッダー（列名）を出力する", value=True)

            if id_col != "なし（予測値のみ）" and raw_test is not None:
                sub_df = pd.DataFrame({id_col: raw_test[id_col].values, pred_col_name: preds})
            else:
                sub_df = pd.DataFrame({pred_col_name: preds})

        st.write("#### 3. 提出前チェック")
        chk1, chk2, chk3, chk4 = st.columns(4)
        chk1.metric("行数", f"{len(sub_df):,}")
        chk2.metric("欠損数", f"{sub_df.isnull().sum().sum():,}")
        chk3.metric("最小値", f"{np.nanmin(preds):.4f}")
        chk4.metric("最大値", f"{np.nanmax(preds):.4f}")

        st.dataframe(sub_df.head(5), use_container_width=True)

        file_name = st.text_input("出力ファイル名", value=f"submission_{st.session_state.last_model_name}.csv")
        csv_data = sub_df.to_csv(index=False, header=include_header).encode("utf-8")

        st.download_button(
            "🚀 Submitファイル (CSV) をダウンロード",
            data=csv_data,
            file_name=file_name,
            mime="text/csv",
            type="primary"
        )
