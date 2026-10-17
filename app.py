import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io

st.set_page_config(page_title="拡張版 CSV前処理アプリ", layout="wide")

# -----------------------------
# 共通処理
# -----------------------------
TARGET_CANDIDATES = ["賃料", "目的変数", "target", "TARGET", "price", "SalePrice"]

@st.cache_data
def read_file_safely(file_bytes, file_name):
    fname_lower = file_name.lower()
    
    # 1. Parquetと推測される場合
    if fname_lower.endswith('.parquet'):
        try:
            return pd.read_parquet(file_bytes)
        except Exception as e:
            st.error(f"Parquet形式としての読み込みに失敗しました: {e}")
            st.stop()
            
    # 2. GZIP圧縮CSVと推測される場合
    elif fname_lower.endswith('.csv.gz') or fname_lower.endswith('.gz'):
        try:
            return pd.read_csv(file_bytes, compression='gzip', low_memory=False)
        except Exception as e:
            st.error(f"GZIP圧縮CSVの読み込みに失敗しました: {e}")
            st.stop()
            
    # 3. CSV形式、またはiPhone等で拡張子が不明になっている場合の汎用処理
    else:
        errors = []
        # まずはCSVとして様々な文字コードで試す
        for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
            try:
                file_bytes.seek(0)
                return pd.read_csv(file_bytes, encoding=enc, low_memory=False)
            except Exception as e:
                errors.append(f"CSV({enc}): {e}")
                
        # CSVとして全滅した場合、ダメ元でParquet形式として読んでみる（iOSの拡張子消失対策）
        try:
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
    if column == "_source_": return None
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

col_ctrl, col_view = st.columns([4, 6], gap="large")

with col_ctrl:
    st.header("⚙️ 設定と実行")
    
    with st.container(border=True):
        st.subheader("1. データ読み込み")
        read_mode = st.radio("読み込みモード", ["単一ファイル", "Train / Test 結合（コンペ用）"], horizontal=True)
        
        if read_mode == "単一ファイル":
            # type制限を外してiPhoneでも全ファイルを選択可能にする
            uploaded_file = st.file_uploader("ファイルをアップロード (.csv, .csv.gz, .parquet)")
            if uploaded_file is not None:
                df = read_file_safely(uploaded_file, uploaded_file.name)
                df["_source_"] = "single"
                st.session_state.df = df
                st.session_state.original_size = uploaded_file.size
        else:
            c1, c2 = st.columns(2)
            with c1: train_file = st.file_uploader("Train")
            with c2: test_file = st.file_uploader("Test")
                
            if train_file is not None and test_file is not None:
                train_df = read_file_safely(train_file, train_file.name)
                test_df = read_file_safely(test_file, test_file.name)
                train_df["_source_"] = "train"
                test_df["_source_"] = "test"
                st.session_state.df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
                st.session_state.original_size = train_file.size + test_file.size

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

        with st.expander("✂️ カラム分割・文字列置換"):
            c1, c2 = st.columns(2)
            with c1:
                split_column = st.selectbox("分割する列", ["(なし)"] + df_cols)
                split_delimiter = st.text_input("区切り文字 (例: / )", value="/", key="split_delim")
            with c2:
                split_prefix = st.text_input("新しい列名の接頭辞")
                split_max_parts = st.number_input("最大分割数", min_value=2, max_value=20, value=3)
            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                replace_column = st.selectbox("文字列を削除する列", ["(なし)"] + df_cols)
            with c4:
                remove_text = st.text_input("削除したい文字列 (例: 円)")

        with st.expander("✨ 特徴量エンジニアリング", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                fe_column = st.selectbox("元になる列", ["(なし)"] + df_cols)
            with c2:
                fe_method = st.selectbox("処理方法", [
                    "何もしない", "文字数カウント", "最初の数値を抽出", 
                    "「徒歩〇分」から最短分数を抽出", "「築〇年」から年数を抽出 (新築=0)", 
                    "キーワード有無フラグ", "ダミー変数化 (One-Hot)", 
                    "マルチラベルダミー化 (区切り文字)", "数値のビン分割 (10分割)"
                ])
            c3, c4, c5 = st.columns(3)
            with c3:
