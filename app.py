import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
import zipfile
import itertools

# グラフの日本語文字化け対策
try:
    import japanize_matplotlib
except ImportError:
    pass

st.set_page_config(page_title="拡張版 CSV前処理アプリ", layout="wide")

# -----------------------------
# 共通処理
# -----------------------------
TARGET_CANDIDATES = ["賃料", "目的変数", "target", "TARGET", "price", "SalePrice", "y", "sales", "kcal"]

def fix_dataframe_for_streamlit(df):
    df = df.copy()
    if df.columns.duplicated().any():
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [
                f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))
            ]
        df.columns = cols
    return df.reset_index(drop=True)

def read_file_safely(file_bytes, file_name):
    fname_lower = file_name.lower()
    if fname_lower.endswith('.parquet'):
        try:
            return pd.read_parquet(file_bytes)
        except Exception as e:
            st.error(f"Parquet形式としての読み込みに失敗しました: {e}")
            st.stop()
    elif fname_lower.endswith('.csv.gz') or fname_lower.endswith('.gz'):
        try:
            return pd.read_csv(file_bytes, compression='gzip', low_memory=False)
        except Exception as e:
            st.error(f"GZIP圧縮CSVの読み込みに失敗しました: {e}")
            st.stop()
    else:
        errors = []
        for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc-jp"]:
            try:
                if hasattr(file_bytes, 'seek'): file_bytes.seek(0)
                return pd.read_csv(file_bytes, encoding=enc, low_memory=False)
            except Exception as e:
                errors.append(f"CSV({enc}): {e}")
        for enc in ["cp932", "utf-8"]:
            try:
                if hasattr(file_bytes, 'seek'): file_bytes.seek(0)
                return pd.read_csv(file_bytes, encoding=enc, errors='replace', low_memory=False)
            except Exception:
                pass
        try:
            if hasattr(file_bytes, 'seek'): file_bytes.seek(0)
            return pd.read_parquet(file_bytes)
        except Exception:
            pass
        st.error("ファイルを読み込めませんでした。\n" + "\n".join(errors))
        st.stop()

def clean_text_value(value):
    if pd.isna(value): return value
    s = str(value)
    s = s.replace("\u3000", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_numeric_columns(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != "_source_"]

def profile_table(df):
    rows = []
    for col in df.columns:
        if col == "_source_": continue
        s = df[col]
        row = {
            "カラム": str(col),
            "データ型": str(s.dtype),
            "欠損数": int(s.isna().sum()),
            "欠損率(%)": round(float(s.isna().mean() * 100), 2) if len(s) else 0,
            "ユニーク数": int(s.nunique(dropna=True)),
            "サンプル値": " | ".join(s.dropna().astype(str).head(3).tolist())
        }
        if pd.api.types.is_numeric_dtype(s):
            row.update({"平均": s.mean(), "最小値": s.min(), "最大値": s.max()})
        else:
            row.update({"平均": np.nan, "最小値": np.nan, "最大値": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)

def make_histogram(df, column):
    if column == "_source_": return None
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty: return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins="auto", color="skyblue", edgecolor="black")
    ax.set_title(f"{column} の分布")
    ax.set_xlabel(str(column))
    ax.set_ylabel("件数")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def convert_df(df, format_type, header=True):
    if format_type == "Parquet形式 (.parquet)":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue(), "parquet", "application/octet-stream"
    elif format_type == "GZIP圧縮CSV (.csv.gz)":
        return df.to_csv(index=False, encoding="utf-8-sig", compression="gzip", header=header).encode("utf-8"), "csv.gz", "application/gzip"
    else:
        return df.to_csv(index=False, encoding="utf-8-sig", header=header).encode("utf-8"), "csv", "text/csv"

# -----------------------------
# メインUI構築
# -----------------------------
st.title("🚀 直感型 データ前処理・特徴量生成アプリ")

if "df" not in st.session_state: st.session_state.df = None
if "processed_df" not in st.session_state: st.session_state.processed_df = None
if "logs" not in st.session_state: st.session_state.logs = []
if "original_size" not in st.session_state: st.session_state.original_size = 0
if "current_file_id" not in st.session_state: st.session_state.current_file_id = None

col_ctrl, col_view = st.columns([4, 6], gap="large")

with col_ctrl:
    c_head1, c_head2 = st.columns([6, 4])
    with c_head1:
        st.header("⚙️ 設定と実行")
    with c_head2:
        if st.session_state.df is not None:
            if st.button("🔄 最初からやり直す", help="元のデータにリセットします", use_container_width=True):
                st.session_state.current_file_id = None
                st.session_state.df = None
                st.session_state.processed_df = None
                st.rerun()

    with st.container(border=True):
        st.subheader("📂 0. データ読み込み")
        read_mode = st.radio("読み込みモード", ["単一ファイル", "Train / Test 個別", "Train / Test まとめてZIP"], horizontal=True)
        
        if read_mode == "単一ファイル":
            uploaded_file = st.file_uploader("ファイルをアップロード (.csv, .csv.gz, .parquet)")
            if uploaded_file is not None:
                fid = f"single_{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.current_file_id != fid:
                    df = read_file_safely(uploaded_file, uploaded_file.name)
                    df["_source_"] = "single"
                    st.session_state.df = fix_dataframe_for_streamlit(df)
                    st.session_state.original_size = uploaded_file.size
                    st.session_state.current_file_id = fid
                    st.session_state.processed_df = None
                    st.session_state.logs = []
                    
        elif read_mode == "Train / Test 個別":
            c1, c2 = st.columns(2)
            with c1: train_file = st.file_uploader("Train")
            with c2: test_file = st.file_uploader("Test")
            if train_file and test_file:
                fid = f"multi_{train_file.name}_{train_file.size}_{test_file.name}_{test_file.size}"
                if st.session_state.current_file_id != fid:
                    train_df = read_file_safely(train_file, train_file.name)
                    test_df = read_file_safely(test_file, test_file.name)
                    train_df["_source_"] = "train"
                    test_df["_source_"] = "test"
                    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
                    st.session_state.df = fix_dataframe_for_streamlit(combined_df)
                    st.session_state.original_size = train_file.size + test_file.size
                    st.session_state.current_file_id = fid
                    st.session_state.processed_df = None
                    st.session_state.logs = []
                    
        elif read_mode == "Train / Test まとめてZIP":
            uploaded_zip = st.file_uploader("ダウンロードしたZIPをアップロード")
            if uploaded_zip is not None:
                fid = f"zip_{uploaded_zip.name}_{uploaded_zip.size}"
                if st.session_state.current_file_id != fid:
                    try:
                        with zipfile.ZipFile(uploaded_zip, 'r') as z:
                            file_list = z.namelist()
                            train_name = next((f for f in file_list if 'train' in f.lower() and not f.endswith('/')), None)
                            test_name = next((f for f in file_list if 'test' in f.lower() and not f.endswith('/')), None)
                            if train_name and test_name:
                                train_df = read_file_safely(io.BytesIO(z.read(train_name)), train_name)
                                test_df = read_file_safely(io.BytesIO(z.read(test_name)), test_name)
                                train_df["_source_"] = "train"
                                test_df["_source_"] = "test"
                                combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
                                st.session_state.df = fix_dataframe_for_streamlit(combined_df)
                                st.session_state.original_size = uploaded_zip.size
                                st.session_state.current_file_id = fid
                                st.session_state.processed_df = None
                                st.session_state.logs = []
                    except Exception as e:
                        st.error(f"ZIPファイルの読み込みに失敗: {e}")

    if st.session_state.df is not None:
        df_cols = [c for c in st.session_state.df.columns.tolist() if c != "_source_"]
        current_num_cols = get_numeric_columns(st.session_state.df) 
        default_target = next((c for c in TARGET_CANDIDATES if c in df_cols), df_cols[0])
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("---")
        st.subheader("🛠 前処理ワークフロー")
        
        target_col = st.selectbox("🎯 目的変数（削除や補完から除外して守る列）", ["(なし)"] + df_cols, index=df_cols.index(default_target)+1 if default_target in df_cols else 0)

        # ----------------------------------------------------------------
        # 🔐 安全ロック用リスト群 (超重要)
        # ----------------------------------------------------------------
        safe_df_cols = [c for c in df_cols if c != target_col]
        safe_num_cols = [c for c in current_num_cols if c != target_col]
        
        # 連続値のみ (IDカラムを除外。四則演算や平均集計にのみ使用許可)
        continuous_num_cols = [c for c in safe_num_cols if not str(c).endswith('_ID')]
        
        if target_col != "(なし)":
            st.success(f"🔒 **【安全ロック作動中】** 目的変数「{target_col}」は、データリーク（過学習）を防ぐため生成メニューから除外されています。")

        # ----------------------------------------------------------------
        # STEP1: 文字列の処理 (分割 / ダミー化 / ID化)
        # ----------------------------------------------------------------
        with st.expander("🪄 STEP1: 複雑な文字列の処理 (分割 / ダミー化 / ID化)", expanded=True):
            tab1, tab2 = st.tabs(["✂️ 文字列のまま複数列に分割", "🔄 カテゴリの数値化 (ID化 / ダミー化)"])
            
            with tab1:
                st.markdown("##### ✂️ 記号や空白で文字をバラバラにして列を分ける")
                c1, c2 = st.columns([5, 5])
                with c1:
                    split_col_step1 = st.selectbox("分割する列", ["(なし)"] + safe_df_cols, key="split_col_step1")
                    split_strategy = st.radio("分割の方式", ["単語ごとに専用列を作成", "左詰めで分割 (最大N列指定)"])
                    if split_strategy == "左詰めで分割 (最大N列指定)":
                        split_max = st.number_input("最大何列に分けるか", min_value=2, max_value=50, value=5)
                        split_sort_mode = st.selectbox("単語のソート方法", ["五十音順", "出現頻度順", "ソートしない"])
                with c2:
                    split_delim = st.text_input("区切り文字", value="自動認識 (/, |, [], 空白, 全角半角)")
                    split_drop_step1 = st.checkbox("処理後に元の列を削除する", value=True, key="split_drop_step1")
            
            with tab2:
                st.markdown("##### 🔢 カテゴリの一括数値ID化 (Label Encoding)")
                le_auto_all = st.checkbox("🪄 すべての文字カラムを全自動で数値IDに変換する", value=False)
                if not le_auto_all:
                    le_columns = st.multiselect("ID化する列を手動で選択", safe_df_cols)
                else:
                    le_columns = []
                le_drop = st.checkbox("ID化の処理後、元の文字カラムを自動で削除する", value=True, key="le_drop")
                
                st.divider()
                st.markdown("##### 🔄 複数キーワードの一括フラグ化")
                c1_2, c2_2 = st.columns([6, 4])
                with c1_2: ml_column = st.selectbox("複雑な文字列が含まれる列を選ぶ", ["(なし)"] + safe_df_cols, key="ml_col")
                with c2_2: ml_delim = st.text_input("区切り文字", value="／", key="ml_delim")
                ml_drop_original = st.checkbox("処理後に元の列を自動で削除する", value=True, key="ml_drop")
                
                st.divider()
                st.markdown("##### 🔄 通常のカテゴリダミー化 (One-Hot)")
                onehot_columns = st.multiselect("ダミー化（0と1に変換）する列を複数選択", safe_df_cols)

        # ----------------------------------------------------------------
        # STEP2: 基本処理
        # ----------------------------------------------------------------
        with st.expander("🛠 STEP2: 基本処理 (欠損値・異常値・文字の微調整)"):
            missing_action = st.radio("欠損値処理", ["変更しない", "数値列を中央値で補完", "欠損行を削除"], horizontal=True)
            outlier_action = st.radio("異常値処理", ["変更しない", "IQRで異常値行を削除"], horizontal=True)
            clean_text = st.checkbox("文字列の不要な空白・改行を整理", value=True)
            duplicate = st.checkbox("完全一致の重複行を削除", value=False)
            
            st.divider()
            st.markdown("##### 📝 セル内文字の微調整 (置換・抽出)")
            c1, c2 = st.columns(2)
            with c1:
                replace_column = st.selectbox("一部の文字を消す列", ["(なし)"] + safe_df_cols, key="replace_col")
                extract_column = st.selectbox("特定の文字だけ残す(抽出)列", ["(なし)"] + safe_df_cols, key="extract_col")
            with c2:
                remove_text = st.text_input("消したい文字 (カンマ区切り可)")
                extract_text = st.text_input("残したい文字 (カンマ区切り可)")

        # ----------------------------------------------------------------
        # STEP3: その他の特徴量作成
        # ----------------------------------------------------------------
        with st.expander("✨ STEP3: その他の特徴量作成 (時系列・数値・集約など)", expanded=True):
            c1, c2 = st.columns([4, 6])
            with c2:
                fe_method = st.selectbox("処理方法を選択", [
                    "何もしない", 
                    "【集約】カテゴリごとの平均値などを算出 (複数選択可)",
                    "【自動生成】数値カラム同士の総当たり割り算 (A ÷ B)", 
                    "【変換】カテゴリを頻度（出現回数）に変換 (Count Encoding)", 
                    "【変換】カテゴリを数値IDに変換 (Label Encoding - 決定木向け)",
                    "【型変換】カテゴリ型 (category) に変換 (LightGBMネイティブ用)",
                    "【変換】数値を任意の幅でグループ化 (例: 10刻みで「〇〇代」)",
                    "【数値計算】2つの列で四則演算 (+, -, *, /)",
                    "【数値計算】対数変換 (Log1p - 歪み補正)",
                    "【変換】数値のビン分割 (10分割)",
                    "【時系列】カテゴリ別の過去実績(y)の平均 (ターゲットエンコーディング)", # 🎯 NEW!
                    "【時系列】過去の値を取得 (ラグ特徴量)",
                    "【時系列】日付列から年・月・日・曜日を抽出",  
                    "【時系列】日付のフォーマット変換", 
                    "【抽出】「徒歩〇分」から最短分数を抽出", 
                    "【抽出】「築〇年」から年数を抽出 (単一の数値に変換)", 
                    "【抽出】「〇年〇ヶ月」を年と月に別々の列に分割",
                    "【抽出】最初の数値を抽出", 
                    "【テキスト】文字数カウント", 
                    "【フラグ】キーワード有無フラグ"
                ])
            
            with c1:
                # 計算系の場合は、第一引数からIDカラムを除外する
                if fe_method in ["【自動生成】数値カラム同士の総当たり割り算 (A ÷ B)", "【数値計算】2つの列で四則演算 (+, -, *, /)"]:
                    allowed_cols = continuous_num_cols
                elif fe_method == "【時系列】過去の値を取得 (ラグ特徴量)":
                    allowed_cols = df_cols # y を選択可能に
                elif fe_method == "【時系列】カテゴリ別の過去実績(y)の平均 (ターゲットエンコーディング)":
                    allowed_cols = safe_df_cols # groupby対象なのでy以外（name_ID等）を選択可能に
                else:
                    allowed_cols = safe_df_cols
                fe_column = st.selectbox("基準となる列（元データ）", ["(なし)"] + allowed_cols)
                
            c3, c4 = st.columns([4, 6])
            with c3:
                fe_prefix = st.text_input("追加する特徴量の接頭辞 (任意)", placeholder="例: feat")
                
            with c4:
                date_format = None; fe_keyword = ""; fe_col2 = None; fe_op = "+"
                fe_num_col = None; fe_aggs = ["mean"]; fe_group_width = 10; fe_group_suffix = "代"
                fe_lag_n = 1; fe_lag_group = "(なし)"; fe_auto_cols = []

                if fe_method == "【時系列】日付のフォーマット変換":
                    date_format = st.selectbox("変換後のフォーマット", ["yyyy-m-d", "yyyy-MM-dd", "yyyy/m/d", "yyyy/MM/dd", "yyyyMMdd"])
                elif fe_method == "【自動生成】数値カラム同士の総当たり割り算 (A ÷ B)":
                    st.info("💡 選択した列のすべての組み合わせで割り算(➗)の特徴量を自動生成します。")
                    st.success("🔒 **【ノイズ防止機能 作動中】** 末尾に「_ID」がつくカテゴリ列は除外されています。")
                    fe_auto_cols = st.multiselect("総当たり計算に使う列を選択 (複数可)", continuous_num_cols, default=continuous_num_cols[:min(5, len(continuous_num_cols))])
                elif fe_method == "【時系列】カテゴリ別の過去実績(y)の平均 (ターゲットエンコーディング)":
                    st.info("💡 リーク（過学習）防止のため、当日のデータを含めずに過去の目的変数(y)の平均を計算します。")
                    fe_lag_n = st.number_input("何回分（何日分）の過去平均を取るか", min_value=1, max_value=365, value=7)
                elif fe_method == "【時系列】過去の値を取得 (ラグ特徴量)":
                    col4_1, col4_2 = st.columns(2)
                    with col4_1: fe_lag_n = st.number_input("何行前の値を取得するか", min_value=1, max_value=365, value=1)
                    with col4_2: fe_lag_group = st.selectbox("グループ化してラグを取る列 (任意)", ["(なし)"] + safe_df_cols)
                elif fe_method == "【変換】数値を任意の幅でグループ化":
                    col4_1, col4_2 = st.columns(2)
                    with col4_1: fe_group_width = st.number_input("区切り幅", min_value=1, value=10)
                    with col4_2: fe_group_suffix = st.text_input("後ろにつける文字", value="代")
                elif fe_method == "【フラグ】キーワード有無フラグ":
                    fe_keyword = st.text_input("キーワード (カンマ区切り)", placeholder="例: カレー, ハンバーグ")
                elif fe_method == "【数値計算】2つの列で四則演算 (+, -, *, /)":
                    col4_1, col4_2 = st.columns([3, 7])
                    with col4_1: fe_op = st.radio("計算記号", ["+", "-", "×", "÷"], horizontal=True)
                    with col4_2: fe_col2 = st.selectbox("計算する2つ目の列", continuous_num_cols, help="IDカラムは除外されています")
                elif fe_method == "【集約】カテゴリごとの平均値などを算出 (複数選択可)":
                    col4_1, col4_2 = st.columns(2)
                    with col4_1:
                        fe_num_col = st.selectbox("集計する数値列 (例: kcal)", ["(なし)"] + continuous_num_cols)
                    with col4_2:
                        fe_aggs = st.multiselect("集計方法", ["mean", "median", "sum", "std", "max", "min", "count"], default=["mean"])

        # ----------------------------------------------------------------
        # STEP4: 大掃除
        # ----------------------------------------------------------------
        with st.expander("🗑 STEP4: 大掃除 (いらない列の削除)", expanded=True):
            st.markdown("##### 🧹 カラム名（列の名前）を指定して一括削除")
            drop_by_name = st.text_input("列名にこの文字が含まれていたらすべて削除 (カンマ区切り可)")
            st.divider()
            st.markdown("##### 📉 マイナーなフラグ列を自動削除")
            min_count = st.number_input("出現件数がこれ以下のダミー列を自動削除", min_value=0, value=0)
            st.divider()
            st.markdown("##### 🗑 手動で選んで削除")
            drop_columns = st.multiselect("リストから選択", safe_df_cols)
            st.divider()
            st.markdown("##### 🎯 数値の範囲でフィルタ (行の削除)")
            c1, c2 = st.columns(2)
            with c1: range_column = st.selectbox("数値フィルタ列 (指定範囲外を削除)", ["(なし)"] + continuous_num_cols)
            with c2: min_value = st.text_input("残す最小値"); max_value = st.text_input("残す最大値")

        st.write("")
        
        # ----------------------------------------------------------------
        # 実行ボタンと処理ロジック
        # ----------------------------------------------------------------
        if st.button("🚀 前処理を実行する", type="primary", use_container_width=True):
            with st.spinner("処理中..."):
                df = st.session_state.df.copy()
                log = []
                target = target_col if target_col != "(なし)" else None

                # -------------------------
                # STEP1: 分割 ＆ フラグ化 ＆ ID化
                # -------------------------
                if split_col_step1 != "(なし)":
                    prefix = f"{split_col_step1}_分割"
                    temp_series = df[split_col_step1].fillna("").astype(str)
                    actual_delim = split_delim
                    if split_delim.startswith("自動認識"): actual_delim = '|||'; temp_series = temp_series.str.replace(r'[／/｜|\[\]［］\s]+', '|||', regex=True).str.strip('|||')
                    elif split_delim == "空白": actual_delim = '|||'; temp_series = temp_series.str.replace(r'\s+', '|||', regex=True).str.strip('|||')

                    if split_strategy.startswith("単語ごとに"):
                        dummies = temp_series.str.get_dummies(sep=actual_delim)
                        dummies = dummies.drop(columns=["", "nan"], errors="ignore")
                        for c in dummies.columns: df[f"{prefix}_{c}"] = dummies[c].map({1: c, 0: np.nan})
                    else:
                        n_parts = int(split_max)
                        split_lists = temp_series.apply(lambda x: [p.strip() for p in str(x).split(actual_delim) if p.strip()] if not pd.isna(x) and x != "" else [])
                        for i in range(n_parts): df[f"{prefix}_{i+1}"] = split_lists.apply(lambda x: x[i] if i < len(x) else np.nan)
                    if split_drop_step1 and split_col_step1 in df.columns: df = df.drop(columns=[split_col_step1])
                    log.append("✂️ 文字列分割を実行しました")

                if ml_column != "(なし)":
                    temp_series = df[ml_column].fillna("").astype(str)
                    actual_delim = ml_delim if ml_delim else "/"
                    if actual_delim == "空白": temp_series = temp_series.str.strip().str.replace(r'\s+', '|', regex=True); actual_delim = '|'
                    dummies = temp_series.str.get_dummies(sep=actual_delim).astype(int)
                    dummies = dummies.drop(columns=["", "nan"], errors="ignore")
                    dummies.columns = [f"{ml_column}_{c}" for c in dummies.columns]
                    df = pd.concat([df, dummies], axis=1)
                    if ml_drop_original and ml_column in df.columns: df = df.drop(columns=[ml_column])
                    log.append("🪄 一括フラグ化を実行しました")

                if onehot_columns:
                    df = pd.get_dummies(df, columns=onehot_columns, dummy_na=False, dtype=int)
                    log.append(f"🔄 One-Hot変換：{', '.join(onehot_columns)} を変換")

                # 一括ID化ロジック
                le_target_cols = set()
                if le_auto_all:
                    text_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
                    le_target_cols.update([c for c in text_cols if c not in [target, "_source_"]])
                if le_columns: le_target_cols.update([c for c in le_columns if c in df.columns])
                le_target_cols = list(le_target_cols)
                
                if le_target_cols:
                    count_le = 0
                    for c in le_target_cols:
                        codes, uniques = pd.factorize(df[c])
                        df[f"{c}_ID"] = np.where(pd.isna(df[c]), np.nan, codes)
                        if le_drop and c in df.columns: df = df.drop(columns=[c])
                        count_le += 1
                    log.append(f"🔢 一括数値ID化：{count_le}列を変換（元データ削除: {'ON' if le_drop else 'OFF'}）")

                # -------------------------
                # STEP2: 基本処理
                # -------------------------
                if clean_text:
                    text_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns if c != "_source_"]
                    for col in text_cols: df[col] = df[col].map(clean_text_value)
                if missing_action == "数値列を中央値で補完":
                    for col in get_numeric_columns(df):
                        if col != target and df[col].isna().any():
                            df[col] = df[col].fillna(df[df["_source_"] == "train"][col].median() if "train" in df["_source_"].values else df[col].median())
                elif missing_action == "欠損行を削除":
                    df = df.dropna(subset=[target]) if target else df.dropna(how='all')
                if duplicate: df = df.drop_duplicates()
                if outlier_action == "IQRで異常値行を削除":
                    for col in get_numeric_columns(df):
                        if col == target: continue
                        s = pd.to_numeric(df[col], errors="coerce").dropna()
                        if len(s) < 4: continue
                        q1, q3 = s.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        df = df[~((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr))]

                if replace_column != "(なし)" and remove_text.strip():
                    for word in [w.strip() for w in remove_text.split(",") if w.strip()]:
                        if replace_column in df.columns: df[replace_column] = df[replace_column].astype(str).str.replace(word, "", regex=False)

                if extract_column != "(なし)" and extract_text.strip():
                    pattern = '|'.join([re.escape(w.strip()) for w in extract_text.split(",") if w.strip()])
                    def extract_match(text):
                        if pd.isna(text): return np.nan
                        matches = re.findall(pattern, str(text))
                        return ",".join(list(dict.fromkeys(matches))) if matches else np.nan
                    if extract_column in df.columns: df[extract_column] = df[extract_column].apply(extract_match)

                # -------------------------
                # STEP3: その他の特徴量作成
                # -------------------------
                if fe_method == "【自動生成】数値カラム同士の総当たり割り算 (A ÷ B)":
                    if len(fe_auto_cols) >= 2:
                        pfx = f"{fe_prefix.strip()}_" if fe_prefix.strip() else ""
                        valid_auto_cols = [c for c in fe_auto_cols if c in df.columns]
                        for c1, c2 in itertools.permutations(valid_auto_cols, 2):
                            s1 = pd.to_numeric(df[c1], errors="coerce")
                            s2 = pd.to_numeric(df[c2], errors="coerce")
                            df[f"{pfx}{c1}÷{c2}"] = np.where(s2 == 0, np.nan, s1 / s2)
                        log.append(f"特徴量追加：{len(valid_auto_cols)}列の総当たり割り算(➗)を実行")

                elif fe_method == "【時系列】カテゴリ別の過去実績(y)の平均 (ターゲットエンコーディング)":
                    if target and target != "(なし)" and fe_column != "(なし)" and fe_column in df.columns:
                        prefix = fe_prefix.strip() or f"{fe_column}"
                        # 🎯 カンニング防止！必ず shift(1) で1つズラして平均をとる
                        df[f"{prefix}_{target}_{fe_lag_n}回過去平均"] = df.groupby(fe_column)[target].transform(lambda x: x.shift(1).rolling(window=fe_lag_n, min_periods=1).mean())
                        log.append(f"特徴量追加：{fe_column}ごとの過去の{target}平均({fe_lag_n}回分)を追加(リーク防止済)")
                    else:
                        st.error("目的変数が選択されていないか、基準列が選ばれていません。")

                elif fe_column != "(なし)" and fe_method != "何もしない" and fe_column in df.columns:
                    source = df[fe_column]
                    prefix = fe_prefix.strip() or f"{fe_column}"
                    
                    if fe_method == "【変換】カテゴリを数値IDに変換 (Label Encoding - 決定木向け)":
                        df[f"{prefix}_ID"] = np.where(pd.isna(source), np.nan, pd.factorize(source)[0])
                    elif fe_method == "【変換】カテゴリを頻度（出現回数）に変換 (Count Encoding)":
                        df[f"{prefix}_頻度"] = source.map(source.value_counts(dropna=False))
                    elif fe_method == "【型変換】カテゴリ型 (category) に変換 (LightGBMネイティブ用)":
                        df[f"{prefix}_cat"] = source.astype("category")
                    elif fe_method == "【変換】数値を任意の幅でグループ化":
                        df[f"{prefix}_{fe_group_width}区切り"] = source.apply(lambda val: f"{int(np.floor(float(val) / fe_group_width) * fe_group_width)}{fe_group_suffix}" if pd.notna(val) else np.nan)
                    elif fe_method == "【時系列】日付列から年・月・日・曜日を抽出":
                        dt_series = pd.to_datetime(source, errors="coerce")
                        df[f"{prefix}_年"] = dt_series.dt.year; df[f"{prefix}_月"] = dt_series.dt.month
                        df[f"{prefix}_日"] = dt_series.dt.day; df[f"{prefix}_曜日"] = dt_series.dt.dayofweek
                    elif fe_method == "【時系列】過去の値を取得 (ラグ特徴量)":
                        if fe_lag_group != "(なし)" and fe_lag_group in df.columns:
                            df[f"{prefix}_{fe_lag_group}別_{fe_lag_n}ラグ"] = df.groupby(fe_lag_group)[fe_column].shift(fe_lag_n)
                        else:
                            df[f"{prefix}_{fe_lag_n}ラグ"] = df[fe_column].shift(fe_lag_n)
                    elif fe_method == "【時系列】日付のフォーマット変換":
                        df[f"{prefix}_フォーマット済"] = pd.to_datetime(source, errors="coerce").dt.strftime("%Y-%m-%d" if "yyyy-MM" in date_format else "%Y/%m/%d")
                    elif fe_method == "【数値計算】2つの列で四則演算 (+, -, *, /)":
                        if fe_col2 and fe_col2 != "(なし)" and fe_col2 in df.columns:
                            s1 = pd.to_numeric(df[fe_column], errors="coerce")
                            s2 = pd.to_numeric(df[fe_col2], errors="coerce")
                            op_name = {"+": "和", "-": "差", "×": "積", "÷": "商"}[fe_op]
                            if fe_op == "+": res = s1 + s2
                            elif fe_op == "-": res = s1 - s2
                            elif fe_op == "×": res = s1 * s2
                            elif fe_op == "÷": res = np.where(s2 == 0, np.nan, s1 / s2)
                            df[f"{prefix}_{op_name}_{fe_col2}"] = res
                    elif fe_method == "【集約】カテゴリごとの平均値などを算出 (複数選択可)":
                        if fe_num_col and fe_num_col != "(なし)" and fe_num_col in df.columns:
                            temp_df = pd.DataFrame({"cat": df[fe_column], "num": pd.to_numeric(df[fe_num_col], errors="coerce"), "src": df.get("_source_", "single")})
                            for agg_func in fe_aggs:
                                agg_map = temp_df[temp_df["src"] == "train"].groupby("cat")["num"].agg(agg_func) if "train" in temp_df["src"].values else temp_df.groupby("cat")["num"].agg(agg_func)
                                df[f"{prefix}_{fe_num_col}_{agg_func}"] = df[fe_column].map(agg_map)
                    elif fe_method == "【数値計算】対数変換 (Log1p - 歪み補正)":
                        df[f"{prefix}_log1p"] = np.log1p(np.clip(pd.to_numeric(source, errors="coerce"), 0, None))
                    elif fe_method == "【フラグ】キーワード有無フラグ":
                        for kw in [k.strip() for k in fe_keyword.split(",") if k.strip()]:
                            df[f"{prefix}_{kw}_有無"] = source.astype(str).str.contains(re.escape(kw)).astype(int)

                # -------------------------
                # STEP4: 大掃除
                # -------------------------
                if drop_by_name.strip():
                    drop_words = [w.strip() for w in drop_by_name.split(",") if w.strip()]
                    to_drop = [col for col in df.columns if col not in [target, "_source_"] and any(w in col for w in drop_words)]
                    if to_drop: df = df.drop(columns=to_drop)

                if min_count > 0:
                    for col in list(df.columns):
                        if col not in [target, "_source_"] and pd.api.types.is_numeric_dtype(df[col]):
                            if set(df[col].dropna().unique()).issubset({0, 1}) and df[col].sum() <= min_count:
                                df = df.drop(columns=[col])

                if drop_columns:
                    to_drop = [c for c in drop_columns if c in df.columns and c not in [target, "_source_"]]
                    if to_drop: df = df.drop(columns=to_drop)

                if range_column != "(なし)":
                    numeric = pd.to_numeric(df[range_column], errors="coerce")
                    mask = pd.Series(True, index=df.index)
                    if min_value.strip(): mask &= numeric >= float(min_value)
                    if max_value.strip(): mask &= numeric <= float(max_value)
                    df = df.loc[mask].copy()

                df = fix_dataframe_for_streamlit(df)
                st.session_state.processed_df = df
                st.session_state.logs = log

with col_view:
    st.header("📊 データビュー")
    
    if st.session_state.df is None:
        st.info("👈 左側のメニューからファイルを読み込んでください。")
    else:
        is_processed = st.session_state.processed_df is not None
        target_df = st.session_state.processed_df if is_processed else st.session_state.df
        num_cols_list = get_numeric_columns(target_df)
        
        view_tabs = st.tabs(["データ確認 ＆ ダウンロード", "プロファイル・分布", "📊 グループ集計 (EDA)", "🔥 相関ヒートマップ", "🌳 特徴量重要度 (AI判定)"])
        
        with view_tabs[0]:
            if is_processed:
                st.success("✅ 前処理が完了しました！")
            else:
                st.info("💡 現在のデータ（前処理前、または引き継ぎ後の状態）です。そのままダウンロードも可能です。")
                
            display_df = target_df.drop(columns=["_source_"], errors="ignore")
            st.markdown(f"**データサイズ: {display_df.shape[0]}行 × {display_df.shape[1]}列**")
            st.dataframe(display_df.head(1000), height=400, use_container_width=True)
            
            if is_processed:
                if st.button("現在の結果をベースに設定を続ける", use_container_width=True):
                    st.session_state.df = target_df
                    st.session_state.processed_df = None
                    st.session_state.logs = []
                    st.rerun()

            st.divider()
            st.markdown("### 📥 データのダウンロード")
            save_format = st.radio("「**保存形式を選択**」", ["通常のCSV (.csv)", "GZIP圧縮CSV (.csv.gz)", "Parquet形式 (.parquet)"], horizontal=True)
            export_header = not st.checkbox("CSVのヘッダーを含めずにダウンロード", value=False)
            
            source_vals = target_df.get("_source_", pd.Series()).unique()
            if "train" in source_vals and "test" in source_vals:
                train_out = target_df[target_df["_source_"] == "train"].drop(columns=["_source_"], errors="ignore")
                test_out = target_df[target_df["_source_"] == "test"].drop(columns=["_source_"], errors="ignore")
                train_data, ext, mime = convert_df(train_out, save_format, header=export_header)
                test_data, _, _ = convert_df(test_out, save_format, header=export_header)
                
                dl_c1, dl_c2, dl_c3 = st.columns(3)
                with dl_c1: st.download_button("TrainデータをDL", train_data, f"preprocessed_train.{ext}", mime, type="secondary", use_container_width=True)
                with dl_c2: st.download_button("TestデータをDL", test_data, f"preprocessed_test.{ext}", mime, type="secondary", use_container_width=True)
                with dl_c3:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        zip_file.writestr(f"preprocessed_train.{ext}", train_data)
                        zip_file.writestr(f"preprocessed_test.{ext}", test_data)
                    st.download_button("ZIPでまとめてDL", zip_buffer.getvalue(), "preprocessed_data.zip", "application/zip", type="primary", use_container_width=True)
            else:
                out_data, ext, mime = convert_df(target_df.drop(columns=["_source_"], errors="ignore"), save_format, header=export_header)
                st.download_button("「**ダウンロード**」", out_data, f"preprocessed_data.{ext}", mime, type="primary")

        with view_tabs[1]:
            st.markdown("**全カラムの基本情報**")
            st.dataframe(profile_table(target_df), use_container_width=True)

        with view_tabs[2]:
            st.markdown("### 🔍 カテゴリごとの統計情報を確認")
            df_cols_clean = [c for c in target_df.columns if c != "_source_"]
            c_eda1, c_eda2 = st.columns(2)
            with c_eda1: grp_col = st.selectbox("グループ化する列 (曜日など)", ["(なし)"] + df_cols_clean)
            with c_eda2: tgt_col = st.selectbox("集計する数値列 (販売数など)", ["(なし)"] + num_cols_list)
            
            if grp_col != "(なし)" and tgt_col != "(なし)":
                try:
                    s_tgt = pd.to_numeric(target_df[tgt_col], errors="coerce")
                    agg_df = s_tgt.groupby(target_df[grp_col]).agg(["count", "sum", "mean", "median", "std"]).round(1)
                    st.dataframe(agg_df, use_container_width=True)
                except Exception as e:
                    pass

        with view_tabs[3]:
            st.markdown("### 🔥 変数間の相関関係 (ヒートマップ)")
            if len(num_cols_list) >= 2:
                corr_cols = st.multiselect("相関を確認する列を選択", num_cols_list, default=num_cols_list[:min(15, len(num_cols_list))])
                if len(corr_cols) >= 2:
                    corr_matrix = target_df[corr_cols].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"shrink": .8})
                    fig.tight_layout()
                    st.pyplot(fig)

        with view_tabs[4]:
            st.markdown("### 🌳 特徴量重要度 (AI判定)")
            if target_col == "(なし)":
                st.warning("👈 左側のメニューで「目的変数」を選択してください。")
            else:
                if st.button("🚀 AIで特徴量重要度を計算する", type="primary", use_container_width=True):
                    with st.spinner("モデルを学習中..."):
                        try:
                            import lightgbm as lgb
                            train_df_model = target_df[target_df["_source_"] == "train"].copy() if "train" in target_df.get("_source_", pd.Series()).values else target_df.copy()
                            train_df_model = train_df_model.dropna(subset=[target_col])
                            y = pd.to_numeric(train_df_model[target_col], errors="coerce")
                            valid_idx = y.notna(); y = y[valid_idx]
                            X = train_df_model[valid_idx].drop(columns=[target_col, "_source_"], errors="ignore")
                            
                            X_features = pd.DataFrame()
                            for c in X.columns:
                                if pd.api.types.is_numeric_dtype(X[c]): X_features[c] = X[c]
                                else:
                                    X_features[c] = pd.factorize(X[c])[0]
                                    X_features[c] = X_features[c].replace(-1, np.nan)
                                    
                            if not X_features.empty:
                                model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                                model.fit(X_features, y)
                                importance_df = pd.DataFrame({"特徴量": X_features.columns, "重要度": model.feature_importances_}).sort_values("重要度", ascending=False)
                                
                                c_imp1, c_imp2 = st.columns([4, 6])
                                with c_imp1: st.dataframe(importance_df, use_container_width=True)
                                with c_imp2:
                                    fig_imp, ax_imp = plt.subplots(figsize=(10, max(5, len(importance_df.head(20))*0.4)))
                                    sns.barplot(data=importance_df.head(20), x="重要度", y="特徴量", ax=ax_imp, palette="viridis")
                                    fig_imp.tight_layout()
                                    st.pyplot(fig_imp)
                                    
                                st.divider()
                                st.markdown("### ✂️ ワンクリック自動整理")
                                keep_n = st.slider("残す上位特徴量の数", min_value=1, max_value=len(importance_df), value=min(20, len(importance_df)))
                                if st.button(f"🗑 上位 {keep_n} 件だけを残して削除", type="primary", use_container_width=True):
                                    safe_keep = set(importance_df.head(keep_n)["特徴量"].tolist() + [target_col, "_source_"])
                                    st.session_state.df = target_df.drop(columns=[c for c in target_df.columns if c not in safe_keep])
                                    st.session_state.processed_df = None
                                    st.rerun()
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")
