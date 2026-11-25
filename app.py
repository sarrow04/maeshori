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

def sync_active_dataset():
    """通常分析モードで加工した内容を元のデータ名へ同期"""
    active_name = st.session_state.get("active_ds_name")
    if active_name and active_name in st.session_state.extra_dfs and st.session_state.train_df is not None:
        st.session_state.extra_dfs[active_name] = st.session_state.train_df

def notify_and_refresh(msg):
    sync_active_dataset()
    save_backup()
    st.session_state.flash_msg = msg
    st.rerun()

def get_all_datasets(current_mode):
    """管理中の全データセットを辞書で返す"""
    datasets = {}
    if current_mode == "コンペ用モード":
        if st.session_state.train_df is not None:
            datasets["train"] = st.session_state.train_df
        if st.session_state.test_df is not None:
            datasets["test"] = st.session_state.test_df
        for name, df_val in st.session_state.extra_dfs.items():
            datasets[name] = df_val
    else:
        for name, df_val in st.session_state.extra_dfs.items():
            datasets[name] = df_val
        # 復元直後などextra_dfsが空でtrain_dfだけある場合のフォールバック
        if not datasets and st.session_state.train_df is not None:
            datasets["restored_data"] = st.session_state.train_df
    return datasets

def render_custom_save_ui(key_prefix, current_mode):
    st.divider()
    st.write("### 💾 現在のデータを名前を付けて保存")
    active_label = st.session_state.active_ds_name if (current_mode == "通常分析モード" and st.session_state.active_ds_name) else "train"
    c1, c2 = st.columns(2)
    with c1:
        tr_name = st.text_input(f"「{active_label}」の保存ファイル名（拡張子不要）", value=f"{active_label}_processed", key=f"{key_prefix}_tr_name")
        b1, b2 = st.columns(2)
        b1.download_button(
            f"📥 {active_label} (CSV)",
            data=st.session_state.train_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{tr_name}.csv",
            mime="text/csv",
            key=f"{key_prefix}_tr_csv"
        )
        b2.download_button(
            f"📥 {active_label} (Parquet)",
            data=st.session_state.train_df.to_parquet(index=False),
            file_name=f"{tr_name}.parquet",
            key=f"{key_prefix}_tr_pq"
        )
    with c2:
        if current_mode == "コンペ用モード" and st.session_state.test_df is not None:
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
def render_live_preview(current_mode):
    datasets = get_all_datasets(current_mode)
    if not datasets:
        return
    if current_mode == "コンペ用モード" and st.session_state.train_df is not None:
        info_txt = f"Train: {st.session_state.train_df.shape[0]:,}行 × {st.session_state.train_df.shape[1]}列"
    elif st.session_state.active_ds_name and st.session_state.active_ds_name in datasets:
        adf = datasets[st.session_state.active_ds_name]
        info_txt = f"{st.session_state.active_ds_name}: {adf.shape[0]:,}行 × {adf.shape[1]}列"
    else:
        info_txt = f"登録データ: {len(datasets)}件"

    label = f"⚡ リアルタイム確認：基本統計量 & 10行プレビュー（{info_txt} / 全{len(datasets)}データ）"
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
    ("active_ds_name", None),
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
render_live_preview(mode)

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
                "分析用データをアップロード（複数ファイル同時選択可・元のファイル名で登録）",
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
                            base_name = os.path.splitext(f.name)[0]
                            ds_name = base_name
                            cnt = 2
                            while ds_name in st.session_state.extra_dfs:
                                ds_name = f"{base_name}_{cnt}"
                                cnt += 1

                            st.session_state.extra_dfs[ds_name] = df_loaded.copy()
                            # 最初のファイルを自動でアクティブ（分析対象）に設定
                            if st.session_state.active_ds_name is None or st.session_state.active_ds_name not in st.session_state.extra_dfs:
                                st.session_state.active_ds_name = ds_name
                                st.session_state.train_df = st.session_state.extra_dfs[ds_name]

                            st.session_state.loaded_normal_files.add(f.name)
                            added_names.append(ds_name)
                        except Exception as e:
                            st.error(f"{f.name} の読み込みエラー: {e}")

                if added_names:
                    notify_and_refresh(f"ファイルを読み込みました: {', '.join(added_names)}")

                if st.button("選択中のファイルを初期状態に再読み込み"):
                    st.session_state.train_df = None
                    st.session_state.extra_dfs = {}
                    st.session_state.active_ds_name = None
                    st.session_state.loaded_normal_files = set()
                    st.rerun()

            # 通常分析モード：分析対象データの切替UI
            if st.session_state.extra_dfs:
                ds_keys = list(st.session_state.extra_dfs.keys())
                cur_idx = ds_keys.index(st.session_state.active_ds_name) if st.session_state.active_ds_name in ds_keys else 0
                selected_active = st.selectbox(
                    "🎯 現在操作するデータ（タブ2〜5のグラフ・前処理・学習の対象）",
                    ds_keys,
                    index=cur_idx,
                    key="normal_active_selector"
                )
                if selected_active != st.session_state.active_ds_name or st.session_state.train_df is None:
                    st.session_state.active_ds_name = selected_active
                    st.session_state.train_df = st.session_state.extra_dfs[selected_active]
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

        datasets = get_all_datasets(mode)

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
            cur_name = st.session_state.active_ds_name if (mode == "通常分析モード" and st.session_state.active_ds_name) else "Train"
            st.divider()
            st.write(f"### カラム概要（型・欠損・ユニーク数：{cur_name}）")
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
            if mode == "コンペ用モード" and extra_name in ["train", "test"]:
                st.error("コンペ用モードでは「train」「test」以外の名前を指定してください。")
            else:
                try:
                    e_df, e_start, e_end = load_file(extra_file)
                    st.session_state.extra_dfs[extra_name] = e_df.copy()
                    if mode == "通常分析モード" and st.session_state.active_ds_name is None:
                        st.session_state.active_ds_name = extra_name
                        st.session_state.train_df = st.session_state.extra_dfs[extra_name]
                    notify_and_refresh(f"データ「{extra_name}」を登録しました（{e_df.shape[0]:,}行 × {e_df.shape[1]}列）")
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")

        st.divider()
        st.subheader("📂 登録中のデータ一覧・個別保存・個別削除")
        datasets = get_all_datasets(mode)
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
                if mode == "コンペ用モード" and sel_ds == "train":
                    st.session_state.train_df = None
                    if os.path.exists(TRAIN_CACHE_PATH):
                        os.remove(TRAIN_CACHE_PATH)
                elif mode == "コンペ用モード" and sel_ds == "test":
                    st.session_state.test_df = None
                    st.session_state.test_raw_df = None
                    st.session_state.test_preds = None
                    if os.path.exists(TEST_CACHE_PATH):
                        os.remove(TEST_CACHE_PATH)
                else:
                    st.session_state.extra_dfs.pop(sel_ds, None)
                    if st.session_state.active_ds_name == sel_ds:
                        if st.session_state.extra_dfs:
                            next_name = next(iter(st.session_state.extra_dfs))
                            st.session_state.active_ds_name = next_name
                            st.session_state.train_df = st.session_state.extra_dfs[next_name]
                        else:
                            st.session_state.active_ds_name = None
                            st.session_state.train_df = None
                notify_and_refresh(f"データ「{sel_ds}」を削除しました。")

    # --- 3. データ結合（横結合・縦結合） ---
    elif t1_mode == "3. データ結合（横結合・縦結合）":
        datasets = get_all_datasets(mode)
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
                                if mode == "コンペ用モード" and left_name == "train":
                                    st.session_state.train_df = merged
                                elif mode == "コンペ用モード" and left_name == "test":
                                    st.session_state.test_df = merged
                                else:
                                    st.session_state.extra_dfs[left_name] = merged
                                    if st.session_state.active_ds_name == left_name:
                                        st.session_state.train_df = merged
                                notify_and_refresh(f"「{left_name}」に「{right_name}」を横結合しました！（{merged.shape[0]:,}行 × {merged.shape[1]}列）")
                            else:
                                st.session_state.extra_dfs[new_ds_name] = merged
                                if mode == "通常分析モード":
                                    st.session_state.active_ds_name = new_ds_name
                                    st.session_state.train_df = merged
                                notify_and_refresh(f"結合データ「{new_ds_name}」を新規登録しました！（{merged.shape[0]:,}行 × {merged.shape[1]}列）")
                        except Exception as e:
                            st.error(f"結合エラー: {e}")

            else:
                sel_concat = st.multiselect("縦に結合するデータを2つ以上選択", list(datasets.keys()), default=list(datasets.keys())[:2], key="concat_sel")
                cc1, cc2 = st.columns(2)
                concat_dest = cc1.radio("結合結果の保存先", ["新しいデータとして登録", "現在のメインデータに上書き"], horizontal=True, key="concat_dest")
                concat_new_name = cc2.text_input("新しいデータ名（新規登録時）", value="concat_data", key="concat_new_name")

                if st.button("🔗 縦結合 (Concat) を実行"):
                    if len(sel_concat) < 2:
                        st.warning("2つ以上のデータを選択してください。")
                    else:
                        try:
                            concated = pd.concat([datasets[n] for n in sel_concat], ignore_index=True)
                            if concat_dest == "現在のメインデータに上書き":
                                st.session_state.train_df = concated
                                notify_and_refresh(f"縦結合したデータでメインデータを上書きしました！（{concated.shape[0]:,}行 × {concated.shape[1]}列）")
                            else:
                                st.session_state.extra_dfs[concat_new_name] = concated
                                if mode == "通常分析モード":
                                    st.session_state.active_ds_name = concat_new_name
                                    st.session_state.train_df = concated
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
                    if mode == "コンペ用モード" and st.session_state.test_df is not None:
                        st.session_state.test_df = st.session_state.test_df.drop(columns=drop_cols, errors="ignore")

                if fill_method != "何もしない（LightGBMに任せる）":
                    num_c = st.session_state.train_df.select_dtypes(include=[np.number]).columns
                    for c in num_c:
                        val = st.session_state.train_df[c].median() if "中央値" in fill_method else (st.session_state.train_df[c].mean() if "平均値" in fill_method else 0)
                        st.session_state.train_df[c] = st.session_state.train_df[c].fillna(val)
                        if mode == "コンペ用モード" and st.session_state.test_df is not None and c in st.session_state.test_df.columns:
                            st.session_state.test_df[c] = st.session_state.test_df[c].fillna(val)
                notify_and_refresh("基本前処理を適用しました！（リアルタイム確認も更新済み）")

        elif prep_mode == "2. 日付・時刻の分解":
            st.write("### 日付列を年・月・日・曜日などに分解")
            dt_col = st.selectbox("日付・時刻の列を選択", all_cols, key="dt_col")
            dt_parts = st.multiselect(
                "作成する項目を選択",
                ["年 (year)", "月 (month)", "日 (day)", "曜日 (dayofweek: 0=月〜6=日)", "土日フラグ (is_weekend)", "時 (hour)"],
                default=["年 (
