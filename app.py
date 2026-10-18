import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import zipfile

st.set_page_config(page_title="拡張版 CSV前処理アプリ", layout="wide")

# -----------------------------
# 共通処理
# -----------------------------
TARGET_CANDIDATES = ["賃料", "目的変数", "target", "TARGET", "price", "SalePrice"]

def fix_dataframe_for_streamlit(df):
    """StreamlitのWebsocket/PyArrowクラッシュを防ぐための自己修復処理"""
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
        for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
            try:
                if hasattr(file_bytes, 'seek'):
                    file_bytes.seek(0)
                return pd.read_csv(file_bytes, encoding=enc, low_memory=False)
            except Exception as e:
                errors.append(f"CSV({enc}): {e}")
                
        try:
            if hasattr(file_bytes, 'seek'):
                file_bytes.seek(0)
            return pd.read_parquet(file_bytes)
        except Exception:
            pass
            
        st.error("ファイルを読み込めませんでした。文字コードやファイル形式が対応していない可能性があります。\n" + "\n".join(errors))
        st.stop()

def clean_text_value(value):
    if pd.isna(value):
        return value
    s = str(value)
    s = s.replace("\u3000", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_numeric_columns(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != "_source_"]

def profile_table(df):
    rows = []
    for col in df.columns:
        if col == "_source_":
            continue
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
    if column == "_source_": 
        return None
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        st.warning("選択した列は数値として扱えません。")
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins="auto", color="skyblue", edgecolor="black")
    ax.set_title(f"{column} の分布")
    ax.set_xlabel(str(column))
    ax.set_ylabel("件数")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def convert_df(df, format_type):
    if format_type == "Parquet形式 (.parquet)":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue(), "parquet", "application/octet-stream"
    elif format_type == "GZIP圧縮CSV (.csv.gz)":
        return df.to_csv(index=False, encoding="utf-8-sig", compression="gzip").encode("utf-8"), "csv.gz", "application/gzip"
    else:
        return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8"), "csv", "text/csv"

# -----------------------------
# メインUI構築
# -----------------------------
st.title("🚀 拡張版 データ前処理・特徴量生成アプリ")

if "df" not in st.session_state:
    st.session_state.df = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "original_size" not in st.session_state:
    st.session_state.original_size = 0
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None

col_ctrl, col_view = st.columns([4, 6], gap="large")

with col_ctrl:
    
    c_head1, c_head2 = st.columns([6, 4])
    with c_head1:
        st.header("⚙️ 設定と実行")
    with c_head2:
        if st.session_state.df is not None:
            if st.button("🔄 最初からやり直す", help="元のアップロードデータにリセットします", use_container_width=True):
                st.session_state.current_file_id = None
                st.session_state.df = None
                st.session_state.processed_df = None
                st.rerun()

    with st.container(border=True):
        st.subheader("1. データ読み込み")
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
            else:
                if st.session_state.current_file_id is not None:
                    st.session_state.current_file_id = None
                    st.session_state.df = None
                    st.session_state.processed_df = None
                    st.rerun()
                    
        elif read_mode == "Train / Test 個別":
            c1, c2 = st.columns(2)
            with c1: 
                train_file = st.file_uploader("Train")
            with c2: 
                test_file = st.file_uploader("Test")
                
            if train_file is not None and test_file is not None:
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
            else:
                if st.session_state.current_file_id is not None:
                    st.session_state.current_file_id = None
                    st.session_state.df = None
                    st.session_state.processed_df = None
                    st.rerun()
                    
        elif read_mode == "Train / Test まとめてZIP":
            uploaded_zip = st.file_uploader("ダウンロードしたZIPファイルをそのままアップロード")
            if uploaded_zip is not None:
                fid = f"zip_{uploaded_zip.name}_{uploaded_zip.size}"
                if st.session_state.current_file_id != fid:
                    try:
                        with zipfile.ZipFile(uploaded_zip, 'r') as z:
                            file_list = z.namelist()
                            train_name = next((f for f in file_list if 'train' in f.lower() and not f.endswith('/')), None)
                            test_name = next((f for f in file_list if 'test' in f.lower() and not f.endswith('/')), None)
                            
                            if train_name and test_name:
                                train_bytes = io.BytesIO(z.read(train_name))
                                test_bytes = io.BytesIO(z.read(test_name))
                                
                                train_df = read_file_safely(train_bytes, train_name)
                                test_df = read_file_safely(test_bytes, test_name)
                                
                                train_df["_source_"] = "train"
                                test_df["_source_"] = "test"
                                combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
                                st.session_state.df = fix_dataframe_for_streamlit(combined_df)
                                st.session_state.original_size = uploaded_zip.size
                                st.session_state.current_file_id = fid
                                st.session_state.processed_df = None
                                st.session_state.logs = []
                            else:
                                st.error("ZIPファイルの中に 'train' と 'test' のデータが見つかりませんでした。")
                    except Exception as e:
                        st.error(f"ZIPファイルの読み込みに失敗しました: {e}")
            else:
                if st.session_state.current_file_id is not None:
                    st.session_state.current_file_id = None
                    st.session_state.df = None
                    st.session_state.processed_df = None
                    st.rerun()

    if st.session_state.df is not None:
        df_cols = [c for c in st.session_state.df.columns.tolist() if c != "_source_"]
        default_target = next((c for c in TARGET_CANDIDATES if c in df_cols), df_cols[0])
        
        st.subheader("2. 前処理の設定")
        target_col = st.selectbox("目的変数（削除や補完の対象外）", ["(なし)"] + df_cols, index=df_cols.index(default_target)+1 if default_target in df_cols else 0)
        
        with st.expander("🛠 基本処理 (欠損・重複・異常値)", expanded=True):
            clean_text = st.checkbox("文字列の不要な空白・改行を整理", value=True)
            duplicate = st.checkbox("完全一致の重複行を削除", value=False)
            missing_action = st.radio("欠損値処理", ["変更しない", "数値列を中央値で補完", "欠損行を削除"], horizontal=True)
            outlier_action = st.radio("異常値処理", ["変更しない", "IQRで異常値行を削除"], horizontal=True)

        with st.expander("✂️ 文字の消去・抽出・カラム分割"):
            c1, c2 = st.columns(2)
            with c1:
                replace_column = st.selectbox("一部の文字を消す列", ["(なし)"] + df_cols, key="replace_col")
                extract_column = st.selectbox("特定の文字だけ残す(抽出)列", ["(なし)"] + df_cols, key="extract_col")
            with c2:
                remove_text = st.text_input("消したい文字 (カンマ区切り可)")
                extract_text = st.text_input("残したい文字 (カンマ区切り可)")
            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                split_column = st.selectbox("分割する列", ["(なし)"] + df_cols)
                split_delimiter = st.text_input("区切り文字 (スペースで分ける場合は「空白」と入力)", value="/", key="split_delim")
            with c4:
                split_prefix = st.text_input("新しい列名の接頭辞")
                split_max_parts = st.number_input("最大分割数", min_value=2, max_value=20, value=3)

        with st.expander("✨ 特徴量エンジニアリング", expanded=True):
            st.markdown("##### 🔄 一括カテゴリダミー化 (One-Hot)")
            onehot_columns = st.multiselect("ダミー化（0/1に変換）する列を複数選択", df_cols, help="選択した列をすべてダミー変数化します（元の列は自動で削除されます）")
            
            st.divider()
            st.markdown("##### 🔍 その他の個別処理")
            c1, c2 = st.columns(2)
            with c1:
                fe_column = st.selectbox("元になる列", ["(なし)"] + df_cols)
            with c2:
                fe_method = st.selectbox("処理方法", [
                    "何もしない", "文字数カウント", "最初の数値を抽出", 
                    "「徒歩〇分」から最短分数を抽出", "「築〇年」から年数を抽出 (新築=0)", 
                    "キーワード有無フラグ", 
                    "マルチラベルダミー化 (区切り文字)", "数値のビン分割 (10分割)"
                ])
            c3, c4, c5 = st.columns(3)
            with c3:
                fe_prefix = st.text_input("接頭辞（任意）")
            with c4:
                fe_keyword = st.text_input("キーワード（フラグ用）")
            with c5:
                fe_delimiter = st.text_input("区切り文字 (スペースは「空白」)", value="/", key="fe_delim")

        with st.expander("🗑 列まるごと削除・数値フィルタ"):
            drop_columns = st.multiselect("🗑 列（カラム）をまるごと削除", df_cols, help="クリックして消したい列を複数選べます")
            st.divider()
            c1, c2 = st.columns(2)
            with c1: 
                range_column = st.selectbox("数値フィルタ列", ["(なし)"] + df_cols)
            with c2: 
                min_value = st.text_input("残す最小値")
                max_value = st.text_input("残す最大値")

        st.write("")
        if st.button("🚀 前処理を実行する", type="primary", use_container_width=True):
            with st.spinner("処理中..."):
                df = st.session_state.df.copy()
                log = []
                target = target_col if target_col != "(なし)" else None

                if clean_text:
                    text_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns if c != "_source_"]
                    for col in text_cols:
                        df[col] = df[col].map(clean_text_value)
                    log.append(f"文字列整理：{len(text_cols)}列の空白・改行を整理")

                if missing_action == "数値列を中央値で補完":
                    for col in get_numeric_columns(df):
                        if col != target and df[col].isna().any():
                            train_median = df[df["_source_"] == "train"][col].median() if "train" in df["_source_"].values else df[col].median()
                            df[col] = df[col].fillna(train_median)
                    log.append("欠損値補完：数値列を補完（Trainの中央値を使用）")
                elif missing_action == "欠損行を削除":
                    if target:
                        before_len = len(df)
                        df = df.dropna(subset=[target]).copy()
                        log.append(f"欠損行削除：目的変数 '{target}' が欠損の {before_len - len(df)} 行のみ削除")
                    else:
                        before_len = len(df)
                        df = df.dropna(how='all').copy()
                        log.append(f"欠損行削除：すべての列が空欄の {before_len - len(df)} 行のみ削除")

                if duplicate:
                    df = df.drop_duplicates().copy()
                    log.append("重複行削除：完全一致する行を削除")

                if outlier_action == "IQRで異常値行を削除":
                    for col in get_numeric_columns(df):
                        if col == target: continue
                        s = pd.to_numeric(df[col], errors="coerce").dropna()
                        if len(s) < 4: continue
                        q1, q3 = s.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        df = df[~((df[col] < low) | (df[col] > high))].copy()
                    log.append("異常値削除：IQR基準で行を削除")

                if range_column != "(なし)":
                    numeric = pd.to_numeric(df[range_column], errors="coerce")
                    mask = pd.Series(True, index=df.index)
                    if min_value.strip(): 
                        try: mask &= numeric >= float(min_value)
                        except Exception: pass
                    if max_value.strip(): 
                        try: mask &= numeric <= float(max_value)
                        except Exception: pass
                    before_len = len(df)
                    df = df.loc[mask].copy()
                    log.append(f"範囲フィルタ：{range_column} で {before_len - len(df)}行を削除")

                if replace_column != "(なし)" and remove_text.strip():
                    remove_words = [w.strip() for w in remove_text.split(",") if w.strip()]
                    for word in remove_words:
                        df[replace_column] = df[replace_column].astype(str).str.replace(word, "", regex=False)
                    log.append(f"セル内文字消去：{replace_column} から '{', '.join(remove_words)}' を消去")

                if extract_column != "(なし)" and extract_text.strip():
                    extract_words = [w.strip() for w in extract_text.split(",") if w.strip()]
                    pattern = '|'.join([re.escape(w) for w in extract_words])
                    
                    def extract_match(text):
                        if pd.isna(text): return np.nan
                        matches = re.findall(pattern, str(text))
                        if matches:
                            seen = set()
                            res = []
                            for m in matches:
                                if m not in seen:
                                    seen.add(m)
                                    res.append(m)
                            return ",".join(res)
                        return np.nan
                    
                    df[extract_column] = df[extract_column].apply(extract_match)
                    log.append(f"文字抽出：{extract_column} に '{', '.join(extract_words)}' のみを残す")

                if split_column != "(なし)" and split_delimiter:
                    prefix = split_prefix.strip() or f"{split_column}_分割"
                    n_parts = int(split_max_parts)
                    temp_series = df[split_column].fillna("").astype(str)
                    
                    if split_delimiter == "空白":
                        split_values = temp_series.str.split(n=n_parts - 1, expand=True)
                    else:
                        if split_delimiter == "/":
                            temp_series = temp_series.str.replace("／", "/")
                        split_values = temp_series.str.split(split_delimiter, n=n_parts - 1, expand=True)
                        
                    for i in range(n_parts):
                        if i < split_values.shape[1]:
                            df[f"{prefix}_{i+1}"] = split_values.iloc[:, i].str.strip().replace("", np.nan)
                        else:
                            df[f"{prefix}_{i+1}"] = np.nan
                    log.append(f"カラム分割：{split_column} を最大 {n_parts} 分割展開")

                if onehot_columns:
                    df = pd.get_dummies(df, columns=onehot_columns, dummy_na=False, dtype=int)
                    log.append(f"一括ダミー化：{', '.join(onehot_columns)} をOne-Hot変換")

                if fe_column != "(なし)" and fe_method != "何もしない":
                    source = df[fe_column]
                    prefix = fe_prefix.strip() or f"{fe_column}"
                    
                    if fe_method == "文字数カウント":
                        df[f"{prefix}_文字数"] = source.astype(str).str.len()
                        log.append(f"特徴量追加：{fe_column} の文字数")
                    elif fe_method == "最初の数値を抽出":
                        extracted = source.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
                        df[f"{prefix}_数値"] = pd.to_numeric(extracted, errors="coerce")
                        log.append(f"特徴量追加：{fe_column} から数値抽出")
                    elif fe_method == "「徒歩〇分」から最短分数を抽出":
                        def get_min_walk(text):
                            if pd.isna(text): return np.nan
                            matches = re.findall(r"徒歩\s*(\d+)\s*分?", str(text))
                            if matches: return min(map(int, matches))
                            return np.nan
                        df[f"{prefix}_最短徒歩分"] = source.apply(get_min_walk)
                        log.append(f"特徴量追加：{fe_column} から徒歩最短分数")
                    elif fe_method == "「築〇年」から年数を抽出 (新築=0)":
                        def get_building_age(text):
                            if pd.isna(text): return np.nan
                            s = str(text)
                            if "新築" in s: return 0.0
                            y_match = re.search(r"(\d+)\s*年", s)
                            m_match = re.search(r"(\d+)\s*ヶ月", s)
                            y = int(y_match.group(1)) if y_match else 0
                            m = int(m_match.group(1)) if m_match else 0
                            if not y_match and not m_match: return np.nan
                            return y + (m / 12.0)
                        df[f"{prefix}_築年数"] = source.apply(get_building_age)
                        log.append(f"特徴量追加：{fe_column} から築年数抽出")
                    elif fe_method == "キーワード有無フラグ":
                        if fe_keyword.strip():
                            df[f"{prefix}_{fe_keyword}_有無"] = source.astype(str).str.contains(re.escape(fe_keyword)).astype(int)
                            log.append(f"特徴量追加：'{fe_keyword}' のフラグ化")
                    elif fe_method == "マルチラベルダミー化 (区切り文字)":
                        delim = fe_delimiter if fe_delimiter else "/"
                        temp_series = source.fillna("").astype(str)
                        
                        if delim == "空白":
                            temp_series = temp_series.str.strip().str.replace(r'\s+', '|', regex=True)
                            actual_delim = '|'
                        else:
                            if delim == "/":
                                temp_series = temp_series.str.replace("／", "/")
                            temp_series = temp_series.str.replace(rf'\s*{re.escape(delim)}\s*', delim, regex=True)
                            actual_delim = delim
                            
                        dummies = temp_series.str.get_dummies(sep=actual_delim).astype(int)
                        
                        if "" in dummies.columns:
                            dummies = dummies.drop(columns=[""])
                        if "nan" in dummies.columns:
                            dummies = dummies.drop(columns=["nan"])
                            
                        if prefix:
                            dummies.columns = [f"{prefix}_{c}" for c in dummies.columns]
                        df = pd.concat([df, dummies], axis=1)
                        log.append(f"特徴量追加：{fe_column} をマルチラベルダミー化（{len(dummies.columns)}列生成）")
                    elif fe_method == "数値のビン分割 (10分割)":
                        num_series = pd.to_numeric(source, errors="coerce")
                        df[f"{prefix}_ビン"] = pd.qcut(num_series, q=10, labels=False, duplicates='drop')
                        log.append(f"特徴量追加：{fe_column} をビン分割")

                if drop_columns:
                    to_drop = [c for c in drop_columns if c in df.columns and c not in [target, "_source_"]]
                    if to_drop:
                        df = df.drop(columns=to_drop)
                        log.append(f"列まるごと削除：{', '.join(to_drop)} を削除")

                df = fix_dataframe_for_streamlit(df)
                st.session_state.processed_df = df
                st.session_state.logs = log

with col_view:
    st.header("📊 データビュー")
    
    if st.session_state.df is None:
        st.info("👈 左側のメニューからファイルをアップロードしてください。")
    else:
        is_processed = st.session_state.processed_df is not None
        target_df = st.session_state.processed_df if is_processed else st.session_state.df
        
        view_tabs = st.tabs(["データ確認 ＆ ダウンロード", "プロファイル・分布"])
        
        with view_tabs[0]:
            if is_processed:
                st.success("✅ 前処理が完了しました！")
                with st.expander("📝 処理ログを確認", expanded=False):
                    for log in st.session_state.logs:
                        st.write(f"- {log}")
            else:
                st.info("💡 現在のデータ（前処理前、または引き継ぎ後の状態）です。そのままダウンロードも可能です。")
                
            display_df = target_df.drop(columns=["_source_"], errors="ignore")
            st.markdown(f"**データサイズ: {display_df.shape[0]}行 × {display_df.shape[1]}列**")
            st.dataframe(display_df.head(1000), height=400, use_container_width=True)
            
            if is_processed:
                st.markdown("### 🔄 さらに前処理を続ける場合")
                st.info("このボタンを押すと、現在の結果をベースにして、新しくできたカラムなどを対象にさらに前処理（列削除など）を続けることができます。")
                if st.button("現在の結果をベースに設定を続ける", use_container_width=True):
                    st.session_state.df = target_df
                    st.session_state.processed_df = None
                    st.session_state.logs = []
                    st.rerun()

            st.divider()
            st.markdown("### 📥 データのダウンロード")
            save_format = st.radio("「**保存形式を選択**」", ["通常のCSV (.csv)", "GZIP圧縮CSV (.csv.gz)", "Parquet形式 (.parquet)"], horizontal=True)
            
            source_vals = target_df.get("_source_", pd.Series()).unique()
            
            if "train" in source_vals and "test" in source_vals:
                train_out = target_df[target_df["_source_"] == "train"].drop(columns=["_source_"], errors="ignore")
                test_out = target_df[target_df["_source_"] == "test"].drop(columns=["_source_"], errors="ignore")
                
                train_data, ext, mime = convert_df(train_out, save_format)
                test_data, _, _ = convert_df(test_out, save_format)
                
                total_new_size = len(train_data) + len(test_data)
                reduction = (1 - total_new_size / st.session_state.original_size) * 100 if st.session_state.original_size > 0 else 0
                
                st.write(f"処理後サイズ合計: {total_new_size / (1024*1024):.2f} MB (削減率: {reduction:.1f}%)")
                
                dl_c1, dl_c2, dl_c3 = st.columns(3)
                with dl_c1:
                    st.download_button("TrainデータをDL", train_data, f"preprocessed_train.{ext}", mime, type="secondary", use_container_width=True)
                with dl_c2:
                    st.download_button("TestデータをDL", test_data, f"preprocessed_test.{ext}", mime, type="secondary", use_container_width=True)
                with dl_c3:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        zip_file.writestr(f"preprocessed_train.{ext}", train_data)
                        zip_file.writestr(f"preprocessed_test.{ext}", test_data)
                    st.download_button("ZIPでまとめてDL", zip_buffer.getvalue(), "preprocessed_data.zip", "application/zip", type="primary", use_container_width=True)
            else:
                out_data, ext, mime = convert_df(target_df.drop(columns=["_source_"], errors="ignore"), save_format)
                reduction = (1 - len(out_data) / st.session_state.original_size) * 100 if st.session_state.original_size > 0 else 0
                
                st.write(f"処理後サイズ: {len(out_data) / (1024*1024):.2f} MB (削減率: {reduction:.1f}%)")
                st.download_button("「**ダウンロード**」", out_data, f"preprocessed_data.{ext}", mime, type="primary")

        with view_tabs[1]:
            st.markdown("**全カラムの基本情報**")
            st.dataframe(profile_table(target_df), use_container_width=True)
            st.divider()
            st.markdown("**ヒストグラム**")
            num_cols = get_numeric_columns(target_df)
            if num_cols:
                hist_col = st.selectbox("分布を確認する列", num_cols)
                fig = make_histogram(target_df, hist_col)
                if fig:
                    st.pyplot(fig)
            else:
                st.info("数値列がありません。")
