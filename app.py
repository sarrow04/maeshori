import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
import zipfile
import itertools

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
            cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
    return df.reset_index(drop=True)

def read_file_safely(file_bytes, file_name):
    fname_lower = file_name.lower()
    if fname_lower.endswith('.parquet'):
        try: return pd.read_parquet(file_bytes)
        except Exception as e: st.error(f"Parquet読込エラー: {e}"); st.stop()
    elif fname_lower.endswith('.csv.gz') or fname_lower.endswith('.gz'):
        try: return pd.read_csv(file_bytes, compression='gzip', low_memory=False)
        except Exception as e: st.error(f"GZIP読込エラー: {e}"); st.stop()
    else:
        for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc-jp"]:
            try:
                if hasattr(file_bytes, 'seek'): file_bytes.seek(0)
                return pd.read_csv(file_bytes, encoding=enc, low_memory=False)
            except Exception: pass
        for enc in ["cp932", "utf-8"]:
            try:
                if hasattr(file_bytes, 'seek'): file_bytes.seek(0)
                return pd.read_csv(file_bytes, encoding=enc, errors='replace', low_memory=False)
            except Exception: pass
        try:
            if hasattr(file_bytes, 'seek'): file_bytes.seek(0)
            return pd.read_parquet(file_bytes)
        except Exception: pass
        st.error("ファイルを読み込めませんでした。")
        st.stop()

def clean_text_value(value):
    if pd.isna(value): return value
    s = str(value).replace("\u3000", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()

def get_numeric_columns(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != "_source_"]

def profile_table(df):
    rows = []
    for col in df.columns:
        if col == "_source_": continue
        s = df[col]
        row = {
            "カラム": str(col), "データ型": str(s.dtype),
            "欠損数": int(s.isna().sum()), "欠損率(%)": round(float(s.isna().mean() * 100), 2) if len(s) else 0,
            "ユニーク数": int(s.nunique(dropna=True)),
            "サンプル値": " | ".join(s.dropna().astype(str).head(3).tolist())
        }
        if pd.api.types.is_numeric_dtype(s): row.update({"平均": s.mean(), "最小値": s.min(), "最大値": s.max()})
        else: row.update({"平均": np.nan, "最小値": np.nan, "最大値": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)

def convert_df(df, format_type, header=True):
    if format_type == "Parquet形式 (.parquet)":
        buffer = io.BytesIO(); df.to_parquet(buffer, index=False)
        return buffer.getvalue(), "parquet", "application/octet-stream"
    elif format_type == "GZIP圧縮CSV (.csv.gz)":
        return df.to_csv(index=False, encoding="utf-8-sig", compression="gzip", header=header).encode("utf-8"), "csv.gz", "application/gzip"
    else:
        return df.to_csv(index=False, encoding="utf-8-sig", header=header).encode("utf-8"), "csv", "text/csv"

# -----------------------------
# メインUI構築
# -----------------------------
st.title("🚀 直感型 データ前処理・特徴量生成アプリ")

for key in ["df", "processed_df", "logs", "original_size", "current_file_id"]:
    if key not in st.session_state: st.session_state[key] = None if key != "logs" else []

col_ctrl, col_view = st.columns([4, 6], gap="large")

with col_ctrl:
    c_head1, c_head2 = st.columns([6, 4])
    with c_head1: st.header("⚙️ 設定と実行")
    with c_head2:
        if st.session_state.df is not None:
            if st.button("🔄 最初からやり直す", use_container_width=True):
                st.session_state.current_file_id = None; st.session_state.df = None; st.session_state.processed_df = None; st.rerun()

    with st.container(border=True):
        st.subheader("📂 0. データ読み込み")
        read_mode = st.radio("読み込みモード", ["単一ファイル", "Train / Test 個別", "Train / Test まとめてZIP"], horizontal=True)
        
        if read_mode == "単一ファイル":
            uploaded_file = st.file_uploader("ファイルをアップロード")
            if uploaded_file:
                fid = f"single_{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.current_file_id != fid:
                    df = read_file_safely(uploaded_file, uploaded_file.name)
                    df["_source_"] = "single"
                    st.session_state.df = fix_dataframe_for_streamlit(df)
                    st.session_state.original_size = uploaded_file.size; st.session_state.current_file_id = fid
                    st.session_state.processed_df = None; st.session_state.logs = []
                    
        elif read_mode == "Train / Test 個別":
            c1, c2 = st.columns(2)
            with c1: train_file = st.file_uploader("Train")
            with c2: test_file = st.file_uploader("Test")
            if train_file and test_file:
                fid = f"multi_{train_file.name}_{train_file.size}_{test_file.name}_{test_file.size}"
                if st.session_state.current_file_id != fid:
                    train_df = read_file_safely(train_file, train_file.name); test_df = read_file_safely(test_file, test_file.name)
                    train_df["_source_"] = "train"; test_df["_source_"] = "test"
                    st.session_state.df = fix_dataframe_for_streamlit(pd.concat([train_df, test_df], axis=0, ignore_index=True))
                    st.session_state.original_size = train_file.size + test_file.size; st.session_state.current_file_id = fid
                    st.session_state.processed_df = None; st.session_state.logs = []

    if st.session_state.df is not None:
        df_cols = [c for c in st.session_state.df.columns.tolist() if c != "_source_"]
        current_num_cols = get_numeric_columns(st.session_state.df) 
        default_target = next((c for c in TARGET_CANDIDATES if c in df_cols), df_cols[0])
        
        st.write("---")
        st.subheader("🛠 前処理ワークフロー")
        target_col = st.selectbox("🎯 目的変数（削除や補完から除外して守る列）", ["(なし)"] + df_cols, index=df_cols.index(default_target)+1 if default_target in df_cols else 0)

        safe_df_cols = [c for c in df_cols if c != target_col]
        safe_num_cols = [c for c in current_num_cols if c != target_col]
        continuous_num_cols = [c for c in safe_num_cols if not str(c).endswith('_ID')]
        
        with st.expander("🪄 STEP1: 複雑な文字列の処理", expanded=True):
            tab1, tab2 = st.tabs(["✂️ 複数列に分割", "🔄 カテゴリの数値化"])
            with tab1:
                c1, c2 = st.columns([5, 5])
                with c1:
                    split_col_step1 = st.selectbox("分割する列", ["(なし)"] + safe_df_cols, key="split_col_step1")
                    split_strategy = st.radio("方式", ["単語ごとに専用列を作成", "左詰めで分割 (最大N列指定)"])
                    if split_strategy.startswith("左詰め"): split_max = st.number_input("最大列数", min_value=2, value=5)
                with c2:
                    split_delim = st.text_input("区切り文字", value="自動認識 (/, |, [], 空白, 全角半角)")
                    split_drop_step1 = st.checkbox("処理後に元の列を削除", value=True)
            with tab2:
                le_auto_all = st.checkbox("🪄 すべての文字カラムを全自動で数値IDに変換", value=False)
                le_columns = [] if le_auto_all else st.multiselect("ID化する列", safe_df_cols)
                le_drop = st.checkbox("元の文字カラムを自動で削除", value=True)
                st.divider()
                c1_2, c2_2 = st.columns([6, 4])
                with c1_2: ml_column = st.selectbox("複雑な文字列が含まれる列を選ぶ", ["(なし)"] + safe_df_cols, key="ml_col")
                with c2_2: ml_delim = st.text_input("区切り文字", value="／")
                ml_drop_original = st.checkbox("処理後に元の列を自動で削除", value=True, key="ml_drop")
                st.divider()
                onehot_columns = st.multiselect("ダミー化（0と1に変換）する列を選択", safe_df_cols)

        with st.expander("🛠 STEP2: 基本処理"):
            missing_action = st.radio("欠損値", ["変更しない", "数値列を中央値で補完", "欠損行を削除"], horizontal=True)
            outlier_action = st.radio("異常値", ["変更しない", "IQRで異常値行を削除"], horizontal=True)
            clean_text = st.checkbox("文字列の不要な空白・改行を整理", value=True)
            duplicate = st.checkbox("完全一致の重複行を削除", value=False)

        with st.expander("✨ STEP3: その他の特徴量作成", expanded=True):
            c1, c2 = st.columns([4, 6])
            with c2:
                fe_method = st.selectbox("処理方法を選択", [
                    "何もしない", 
                    "【集約】カテゴリごとの平均値などを算出 (複数選択可)",
                    "【自動生成】数値カラム同士の総当たり割り算 (A ÷ B)", 
                    "【変換】カテゴリを頻度（出現回数）に変換 (Count Encoding)", 
                    "【変換】カテゴリを数値IDに変換 (Label Encoding - 決定木向け)",
                    "【型変換】カテゴリ型 (category) に変換 (LightGBMネイティブ用)",
                    "【変換】数値を任意の幅でグループ化 (例: 10刻みで「〇〇代」)", # ★メニュー名
                    "【数値計算】2つの列で四則演算 (+, -, *, /)",
                    "【数値計算】対数変換 (Log1p - 歪み補正)",
                    "【変換】数値のビン分割 (10分割)", # ★復活
                    "【時系列】カテゴリ別の過去実績(y)の平均 (ターゲットエンコーディング)",
                    "【時系列】過去の値を取得 (ラグ特徴量)",
                    "【時系列】日付列から年・月・日・曜日を抽出",  
                    "【時系列】日付のフォーマット変換", 
                    "【抽出】「徒歩〇分」から最短分数を抽出", # ★復活
                    "【抽出】「築〇年」から年数を抽出 (単一の数値に変換)", # ★復活
                    "【抽出】「〇年〇ヶ月」を年と月に別々の列に分割", # ★復活
                    "【抽出】最初の数値を抽出", # ★復活
                    "【テキスト】文字数カウント", # ★復活
                    "【フラグ】キーワード有無フラグ"
                ])
            
            with c1:
                if fe_method in ["【自動生成】数値カラム同士の総当たり割り算 (A ÷ B)", "【数値計算】2つの列で四則演算 (+, -, *, /)"]: allowed_cols = continuous_num_cols
                elif fe_method == "【時系列】過去の値を取得 (ラグ特徴量)": allowed_cols = df_cols 
                elif fe_method == "【時系列】カテゴリ別の過去実績(y)の平均 (ターゲットエンコーディング)": allowed_cols = safe_df_cols
                else: allowed_cols = safe_df_cols
                fe_column = st.selectbox("基準となる列（元データ）", ["(なし)"] + allowed_cols)
                
            c3, c4 = st.columns([4, 6])
            with c3: fe_prefix = st.text_input("追加する特徴量の接頭辞 (任意)")
            with c4:
                date_format = None; fe_keyword = ""; fe_col2 = None; fe_op = "+"
                fe_num_col = None; fe_aggs = ["mean"]; fe_group_width = 10; fe_group_suffix = "代"
                fe_lag_n = 1; fe_lag_group = "(なし)"; fe_auto_cols = []

                if fe_method == "【時系列】日付のフォーマット変換": date_format = st.selectbox("フォーマット", ["yyyy-m-d", "yyyy-MM-dd", "yyyy/m/d", "yyyy/MM/dd", "yyyyMMdd"])
                elif fe_method == "【自動生成】数値カラム同士の総当たり割り算 (A ÷ B)": fe_auto_cols = st.multiselect("対象", continuous_num_cols, default=continuous_num_cols[:min(5, len(continuous_num_cols))])
                elif fe_method == "【時系列】カテゴリ別の過去実績(y)の平均 (ターゲットエンコーディング)": fe_lag_n = st.number_input("何回分の過去平均を取るか", min_value=1, value=7)
                elif fe_method == "【時系列】過去の値を取得 (ラグ特徴量)":
                    col4_1, col4_2 = st.columns(2)
                    with col4_1: fe_lag_n = st.number_input("何行前", min_value=1, value=1)
                    with col4_2: fe_lag_group = st.selectbox("グループ列", ["(なし)"] + safe_df_cols)
                elif fe_method == "【変換】数値を任意の幅でグループ化 (例: 10刻みで「〇〇代」)":
                    col4_1, col4_2 = st.columns(2)
                    with col4_1: fe_group_width = st.number_input("区切り幅", min_value=1, value=10)
                    with col4_2: fe_group_suffix = st.text_input("後ろにつける文字", value="代")
                elif fe_method == "【フラグ】キーワード有無フラグ": fe_keyword = st.text_input("キーワード (カンマ区切り)")
                elif fe_method == "【数値計算】2つの列で四則演算 (+, -, *, /)":
                    col4_1, col4_2 = st.columns([3, 7])
                    with col4_1: fe_op = st.radio("計算", ["+", "-", "×", "÷"], horizontal=True)
                    with col4_2: fe_col2 = st.selectbox("2つ目の列", continuous_num_cols)
                elif fe_method == "【集約】カテゴリごとの平均値などを算出 (複数選択可)":
                    col4_1, col4_2 = st.columns(2)
                    with col4_1: fe_num_col = st.selectbox("数値列", ["(なし)"] + continuous_num_cols)
                    with col4_2: fe_aggs = st.multiselect("方法", ["mean", "median", "sum", "std", "max", "min", "count"], default=["mean"])

        with st.expander("🗑 STEP4: 大掃除 (いらない列の削除)"):
            drop_by_name = st.text_input("列名にこの文字が含まれていたら削除 (カンマ区切り)")
            min_count = st.number_input("出現件数がこれ以下のダミー列を削除", min_value=0, value=0)
            drop_columns = st.multiselect("手動で選んで削除", safe_df_cols)

        if st.button("🚀 前処理を実行する", type="primary", use_container_width=True):
            with st.spinner("処理中..."):
                df = st.session_state.df.copy()
                log = []
                target = target_col if target_col != "(なし)" else None

                if split_col_step1 != "(なし)":
                    prefix = f"{split_col_step1}_分割"
                    temp_s = df[split_col_step1].fillna("").astype(str)
                    actual_delim = '|||' if split_delim.startswith("自動認識") or split_delim == "空白" else split_delim
                    if split_delim.startswith("自動認識"): temp_s = temp_s.str.replace(r'[／/｜|\[\]［］\s]+', '|||', regex=True).str.strip('|||')
                    elif split_delim == "空白": temp_s = temp_s.str.replace(r'\s+', '|||', regex=True).str.strip('|||')
                    if split_strategy.startswith("単語ごとに"):
                        dum = temp_s.str.get_dummies(sep=actual_delim).drop(columns=["", "nan"], errors="ignore")
                        for c in dum.columns: df[f"{prefix}_{c}"] = dum[c].map({1: c, 0: np.nan})
                    else:
                        lst = temp_s.apply(lambda x: [p.strip() for p in str(x).split(actual_delim) if p.strip()] if pd.notna(x) and x != "" else [])
                        for i in range(int(split_max)): df[f"{prefix}_{i+1}"] = lst.apply(lambda x: x[i] if i < len(x) else np.nan)
                    if split_drop_step1: df = df.drop(columns=[split_col_step1])

                if ml_column != "(なし)":
                    temp_s = df[ml_column].fillna("").astype(str)
                    delim = '|' if ml_delim == "空白" else (ml_delim if ml_delim else "/")
                    if delim == '|': temp_s = temp_s.str.strip().str.replace(r'\s+', '|', regex=True)
                    dum = temp_s.str.get_dummies(sep=delim).astype(int).drop(columns=["", "nan"], errors="ignore")
                    dum.columns = [f"{ml_column}_{c}" for c in dum.columns]
                    df = pd.concat([df, dum], axis=1)
                    if ml_drop_original: df = df.drop(columns=[ml_column])

                if onehot_columns: df = pd.get_dummies(df, columns=onehot_columns, dummy_na=False, dtype=int)

                le_target_cols = set([c for c in df.select_dtypes(include=["object", "string", "category"]).columns.tolist() if c not in [target, "_source_"]]) if le_auto_all else set(le_columns)
                for c in le_target_cols:
                    if c in df.columns:
                        df[f"{c}_ID"] = np.where(pd.isna(df[c]), np.nan, pd.factorize(df[c])[0])
                        if le_drop: df = df.drop(columns=[c])

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
                        if len(s) >= 4:
                            q1, q3 = s.quantile([0.25, 0.75])
                            iqr = q3 - q1
                            df = df[~((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr))]

                # 🌟 【完全復活】STEP3 の実行ロジック
                if fe_method == "【自動生成】数値カラム同士の総当たり割り算 (A ÷ B)":
                    pfx = f"{fe_prefix.strip()}_" if fe_prefix.strip() else ""
                    for c1, c2 in itertools.permutations([c for c in fe_auto_cols if c in df.columns], 2):
                        s1 = pd.to_numeric(df[c1], errors="coerce"); s2 = pd.to_numeric(df[c2], errors="coerce")
                        df[f"{pfx}{c1}÷{c2}"] = np.where(s2 == 0, np.nan, s1 / s2)

                elif fe_method == "【時系列】カテゴリ別の過去実績(y)の平均 (ターゲットエンコーディング)":
                    if target and target != "(なし)" and fe_column != "(なし)" and fe_column in df.columns:
                        pfx = fe_prefix.strip() or f"{fe_column}"
                        df[f"{pfx}_{target}_{fe_lag_n}回過去平均"] = df.groupby(fe_column)[target].transform(lambda x: x.shift(1).rolling(window=fe_lag_n, min_periods=1).mean())

                elif fe_column != "(なし)" and fe_method != "何もしない" and fe_column in df.columns:
                    source = df[fe_column]
                    prefix = fe_prefix.strip() or f"{fe_column}"
                    
                    if fe_method == "【変換】カテゴリを数値IDに変換 (Label Encoding - 決定木向け)":
                        df[f"{prefix}_ID"] = np.where(pd.isna(source), np.nan, pd.factorize(source)[0])
                    elif fe_method == "【変換】カテゴリを頻度（出現回数）に変換 (Count Encoding)":
                        df[f"{prefix}_頻度"] = source.map(source.value_counts(dropna=False))
                    elif fe_method == "【型変換】カテゴリ型 (category) に変換 (LightGBMネイティブ用)":
                        df[f"{prefix}_cat"] = source.astype("category")
                        
                    # ★ 修正箇所：数値を安全に変換してから計算
                    elif fe_method == "【変換】数値を任意の幅でグループ化 (例: 10刻みで「〇〇代」)":
                        num_s = pd.to_numeric(source, errors="coerce")
                        df[f"{prefix}_{fe_group_width}区切り"] = num_s.apply(
                            lambda val: f"{int(np.floor(val / fe_group_width) * fe_group_width)}{fe_group_suffix}" if pd.notna(val) else np.nan
                        )
                        
                    # ★ 復活箇所：ビン分割
                    elif fe_method == "【変換】数値のビン分割 (10分割)":
                        num_s = pd.to_numeric(source, errors="coerce")
                        df[f"{prefix}_ビン"] = pd.qcut(num_s, q=10, labels=False, duplicates='drop')
                        
                    elif fe_method == "【時系列】日付列から年・月・日・曜日を抽出":
                        dt_s = pd.to_datetime(source, errors="coerce")
                        df[f"{prefix}_年"] = dt_s.dt.year; df[f"{prefix}_月"] = dt_s.dt.month
                        df[f"{prefix}_日"] = dt_s.dt.day; df[f"{prefix}_曜日"] = dt_s.dt.dayofweek
                        
                    elif fe_method == "【時系列】過去の値を取得 (ラグ特徴量)":
                        if fe_lag_group != "(なし)" and fe_lag_group in df.columns:
                            df[f"{prefix}_{fe_lag_group}別_{fe_lag_n}ラグ"] = df.groupby(fe_lag_group)[fe_column].shift(fe_lag_n)
                        else:
                            df[f"{prefix}_{fe_lag_n}ラグ"] = df[fe_column].shift(fe_lag_n)
                            
                    elif fe_method == "【時系列】日付のフォーマット変換":
                        df[f"{prefix}_フォーマット済"] = pd.to_datetime(source, errors="coerce").dt.strftime("%Y-%m-%d" if "yyyy-MM" in date_format else "%Y/%m/%d")
                        
                    elif fe_method == "【数値計算】2つの列で四則演算 (+, -, *, /)":
                        if fe_col2 and fe_col2 != "(なし)" and fe_col2 in df.columns:
                            s1 = pd.to_numeric(df[fe_column], errors="coerce"); s2 = pd.to_numeric(df[fe_col2], errors="coerce")
                            if fe_op == "+": df[f"{prefix}_和_{fe_col2}"] = s1 + s2
                            elif fe_op == "-": df[f"{prefix}_差_{fe_col2}"] = s1 - s2
                            elif fe_op == "×": df[f"{prefix}_積_{fe_col2}"] = s1 * s2
                            elif fe_op == "÷": df[f"{prefix}_商_{fe_col2}"] = np.where(s2 == 0, np.nan, s1 / s2)
                            
                    elif fe_method == "【集約】カテゴリごとの平均値などを算出 (複数選択可)":
                        if fe_num_col and fe_num_col != "(なし)" and fe_num_col in df.columns:
                            tmp = pd.DataFrame({"cat": df[fe_column], "num": pd.to_numeric(df[fe_num_col], errors="coerce"), "src": df.get("_source_", "single")})
                            for agg in fe_aggs:
                                df[f"{prefix}_{fe_num_col}_{agg}"] = df[fe_column].map(tmp[tmp["src"] == "train"].groupby("cat")["num"].agg(agg) if "train" in tmp["src"].values else tmp.groupby("cat")["num"].agg(agg))
                                
                    elif fe_method == "【数値計算】対数変換 (Log1p - 歪み補正)":
                        df[f"{prefix}_log1p"] = np.log1p(np.clip(pd.to_numeric(source, errors="coerce"), 0, None))
                        
                    elif fe_method == "【フラグ】キーワード有無フラグ":
                        for kw in [k.strip() for k in fe_keyword.split(",") if k.strip()]:
                            df[f"{prefix}_{kw}_有無"] = source.astype(str).str.contains(re.escape(kw)).astype(int)

                    # ★ 復活箇所：抽出系メニュー
                    elif fe_method == "【抽出】「徒歩〇分」から最短分数を抽出":
                        df[f"{prefix}_最短徒歩分"] = source.apply(lambda x: min(map(int, re.findall(r"徒歩\s*(\d+)\s*分?", str(x)))) if pd.notna(x) and re.findall(r"徒歩\s*(\d+)\s*分?", str(x)) else np.nan)

                    elif fe_method == "【抽出】「築〇年」から年数を抽出 (単一の数値に変換)":
                        def get_age(s):
                            if pd.isna(s): return np.nan
                            s = str(s)
                            if "新築" in s: return 0.0
                            y_m = re.search(r"(\d+)\s*年", s); m_m = re.search(r"(\d+)\s*[ヶ箇カかケヵ]?月", s)
                            if not y_m and not m_m: return np.nan
                            return (int(y_m.group(1)) if y_m else 0) + ((int(m_m.group(1)) if m_m else 0) / 12.0)
                        df[f"{prefix}_築年数"] = source.apply(get_age)

                    elif fe_method == "【抽出】「〇年〇ヶ月」を年と月に別々の列に分割":
                        df[f"{prefix}_年"] = source.apply(lambda x: 0.0 if "新築" in str(x) else (float(re.search(r"(\d+)\s*年", str(x)).group(1)) if re.search(r"(\d+)\s*年", str(x)) else (0.0 if re.search(r"(\d+)\s*[ヶ箇カかケヵ]?月", str(x)) else np.nan)) if pd.notna(x) else np.nan)
                        df[f"{prefix}_ヶ月"] = source.apply(lambda x: 0.0 if "新築" in str(x) else (float(re.search(r"(\d+)\s*[ヶ箇カかケヵ]?月", str(x)).group(1)) if re.search(r"(\d+)\s*[ヶ箇カかケヵ]?月", str(x)) else (0.0 if re.search(r"(\d+)\s*年", str(x)) else np.nan)) if pd.notna(x) else np.nan)

                    elif fe_method == "【抽出】最初の数値を抽出":
                        df[f"{prefix}_数値"] = pd.to_numeric(source.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")

                    elif fe_method == "【テキスト】文字数カウント":
                        df[f"{prefix}_文字数"] = source.astype(str).str.len()

                if drop_by_name.strip():
                    to_drop = [col for col in df.columns if col not in [target, "_source_"] and any(w in col for w in [w.strip() for w in drop_by_name.split(",") if w.strip()])]
                    if to_drop: df = df.drop(columns=to_drop)
                if min_count > 0:
                    for col in list(df.columns):
                        if col not in [target, "_source_"] and pd.api.types.is_numeric_dtype(df[col]):
                            if set(df[col].dropna().unique()).issubset({0, 1}) and df[col].sum() <= min_count: df = df.drop(columns=[col])
                if drop_columns:
                    to_drop = [c for c in drop_columns if c in df.columns and c not in [target, "_source_"]]
                    if to_drop: df = df.drop(columns=to_drop)

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
            if is_processed: st.success("✅ 前処理が完了しました！")
            
            display_df = target_df.drop(columns=["_source_"], errors="ignore")
            st.markdown(f"**データサイズ: {display_df.shape[0]}行 × {display_df.shape[1]}列**")
            st.dataframe(display_df.head(1000), height=400, use_container_width=True)
            
            if is_processed:
                if st.button("現在の結果をベースに設定を続ける", use_container_width=True):
                    st.session_state.df = target_df; st.session_state.processed_df = None; st.session_state.logs = []; st.rerun()

            st.divider()
            save_format = st.radio("「**保存形式を選択**」", ["通常のCSV (.csv)", "GZIP圧縮CSV (.csv.gz)", "Parquet形式 (.parquet)"], horizontal=True)
            export_header = not st.checkbox("CSVのヘッダーを含めずにダウンロード", value=False)
            
            source_vals = target_df.get("_source_", pd.Series()).unique()
            if "train" in source_vals and "test" in source_vals:
                train_data, ext, mime = convert_df(target_df[target_df["_source_"] == "train"].drop(columns=["_source_"], errors="ignore"), save_format, export_header)
                test_data, _, _ = convert_df(target_df[target_df["_source_"] == "test"].drop(columns=["_source_"], errors="ignore"), save_format, export_header)
                
                dl_c1, dl_c2, dl_c3 = st.columns(3)
                with dl_c1: st.download_button("TrainデータをDL", train_data, f"preprocessed_train.{ext}", mime, use_container_width=True)
                with dl_c2: st.download_button("TestデータをDL", test_data, f"preprocessed_test.{ext}", mime, use_container_width=True)
                with dl_c3:
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
                        z.writestr(f"preprocessed_train.{ext}", train_data); z.writestr(f"preprocessed_test.{ext}", test_data)
                    st.download_button("ZIPでまとめてDL", zip_buf.getvalue(), "preprocessed_data.zip", "application/zip", use_container_width=True)
            else:
                out_data, ext, mime = convert_df(target_df.drop(columns=["_source_"], errors="ignore"), save_format, export_header)
                st.download_button("「**ダウンロード**」", out_data, f"preprocessed_data.{ext}", mime, type="primary")

        with view_tabs[1]:
            st.dataframe(profile_table(target_df), use_container_width=True)

        with view_tabs[2]:
            df_cols_clean = [c for c in target_df.columns if c != "_source_"]
            c_eda1, c_eda2 = st.columns(2)
            with c_eda1: grp_col = st.selectbox("グループ化する列 (曜日など)", ["(なし)"] + df_cols_clean)
            with c_eda2: tgt_col = st.selectbox("集計する数値列 (販売数など)", ["(なし)"] + num_cols_list)
            
            if grp_col != "(なし)" and tgt_col != "(なし)":
                try: st.dataframe(pd.to_numeric(target_df[tgt_col], errors="coerce").groupby(target_df[grp_col]).agg(["count", "sum", "mean", "median", "std"]).round(1), use_container_width=True)
                except Exception: pass

        with view_tabs[3]:
            if len(num_cols_list) >= 2:
                corr_cols = st.multiselect("相関を確認する列", num_cols_list, default=num_cols_list[:min(15, len(num_cols_list))])
                if len(corr_cols) >= 2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(target_df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"shrink": .8})
                    st.pyplot(fig)

        with view_tabs[4]:
            if target_col == "(なし)": st.warning("👈 左側のメニューで「目的変数」を選択してください。")
            else:
                if st.button("🚀 AIで特徴量重要度を計算する", type="primary", use_container_width=True):
                    with st.spinner("モデルを学習中..."):
                        try:
                            import lightgbm as lgb
                            tr_df = target_df[target_df["_source_"] == "train"].copy() if "train" in target_df.get("_source_", pd.Series()).values else target_df.copy()
                            tr_df = tr_df.dropna(subset=[target_col])
                            y = pd.to_numeric(tr_df[target_col], errors="coerce")
                            valid_idx = y.notna(); y = y[valid_idx]
                            X = tr_df[valid_idx].drop(columns=[target_col, "_source_"], errors="ignore")
                            
                            X_features = pd.DataFrame()
                            for c in X.columns:
                                if pd.api.types.is_numeric_dtype(X[c]): X_features[c] = X[c]
                                else: X_features[c] = pd.factorize(X[c])[0].astype(float); X_features[c] = X_features[c].replace(-1.0, np.nan)
                                    
                            if not X_features.empty:
                                model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                                model.fit(X_features, y)
                                imp_df = pd.DataFrame({"特徴量": X_features.columns, "重要度": model.feature_importances_}).sort_values("重要度", ascending=False)
                                
                                c_imp1, c_imp2 = st.columns([4, 6])
                                with c_imp1: st.dataframe(imp_df, use_container_width=True)
                                with c_imp2:
                                    fig_imp, ax_imp = plt.subplots(figsize=(10, max(5, len(imp_df.head(20))*0.4)))
                                    sns.barplot(data=imp_df.head(20), x="重要度", y="特徴量", ax=ax_imp, palette="viridis")
                                    st.pyplot(fig_imp)
                                    
                                st.divider()
                                keep_n = st.slider("残す上位特徴量の数", min_value=1, max_value=len(imp_df), value=min(20, len(imp_df)))
                                if st.button(f"🗑 上位 {keep_n} 件だけを残して削除", type="primary", use_container_width=True):
                                    safe_keep = set(imp_df.head(keep_n)["特徴量"].tolist() + [target_col, "_source_"])
                                    st.session_state.df = target_df.drop(columns=[c for c in target_df.columns if c not in safe_keep])
                                    st.session_state.processed_df = None; st.rerun()
                        except Exception as e: st.error(f"エラーが発生しました: {e}")
