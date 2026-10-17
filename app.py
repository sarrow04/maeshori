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
def read_csv_safely(file_bytes):
    """複数エンコーディングを試してCSVを読み込む"""
    errors = []
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            # バイトデータを先頭に戻して読み込む
            file_bytes.seek(0)
            return pd.read_csv(file_bytes, encoding=enc, low_memory=False)
        except Exception as e:
            errors.append(f"{enc}: {e}")
    st.error("CSVを読み込めませんでした。\n" + "\n".join(errors))
    st.stop()

def clean_text_value(value):
    if pd.isna(value):
        return value
    s = str(value)
    s = s.replace("\u3000", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def profile_table(df):
    rows = []
    for col in df.columns:
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

# -----------------------------
# メインUI構築
# -----------------------------
st.title("🚀 拡張版 CSV前処理・特徴量生成アプリ")
st.markdown("Streamlit版: CSVを確認し、条件を選んで前処理します。元データは上書きしません。")

# セッションステートの初期化
if "df" not in st.session_state:
    st.session_state.df = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "logs" not in st.session_state:
    st.session_state.logs = []

tab1, tab2, tab3 = st.tabs(["1. データ読み込み・確認", "2. 前処理・特徴量エンジニアリング", "3. 結果確認・ダウンロード"])

with tab1:
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df = read_csv_safely(uploaded_file)
        st.success(f"読み込み完了: {st.session_state.df.shape[0]}行 × {st.session_state.df.shape[1]}列")
        
        st.subheader("データプロファイル")
        st.dataframe(profile_table(st.session_state.df), use_container_width=True)
        
        st.subheader("ヒストグラム")
        num_cols = get_numeric_columns(st.session_state.df)
        if num_cols:
            hist_col = st.selectbox("分布を確認する列を選択", num_cols)
            if st.button("ヒストグラムを描画"):
                fig = make_histogram(st.session_state.df, hist_col)
                if fig:
                    st.pyplot(fig)
        else:
            st.info("数値列がありません。")

with tab2:
    if st.session_state.df is not None:
        df_cols = st.session_state.df.columns.tolist()
        default_target = next((c for c in TARGET_CANDIDATES if c in df_cols), df_cols[0])
        
        target_col = st.selectbox("目的変数（削除や補完の対象外にします）", ["(なし)"] + df_cols, index=df_cols.index(default_target)+1 if default_target in df_cols else 0)
        
        with st.expander("🛠 基本処理 (欠損・重複・異常値)", expanded=True):
            clean_text = st.checkbox("文字列の不要な空白・改行を整理", value=True)
            duplicate = st.checkbox("完全一致の重複行を削除", value=False)
            missing_action = st.radio("欠損値処理", ["変更しない", "数値列を中央値で補完", "欠損行を削除"], horizontal=True)
            outlier_action = st.radio("異常値処理", ["変更しない", "IQRで異常値行を削除"], horizontal=True)

        with st.expander("✂️ カラム分割・文字列置換"):
            st.markdown("任意の文字（ / や , など）を指定してカラムを複数に分割できます。")
            col1, col2 = st.columns(2)
            with col1:
                split_column = st.selectbox("分割する列", ["(なし)"] + df_cols)
                split_delimiter = st.text_input("区切り文字 (例: / )", value="/")
            with col2:
                split_prefix = st.text_input("新しい列名の接頭辞")
                split_max_parts = st.number_input("最大分割数", min_value=2, max_value=20, value=3)
            
            st.divider()
            col3, col4 = st.columns(2)
            with col3:
                replace_column = st.selectbox("文字列を削除する列", ["(なし)"] + df_cols)
            with col4:
                remove_text = st.text_input("削除したい文字列 (例: 円, 東京都)")

        with st.expander("✨ 特徴量エンジニアリング"):
            col1, col2 = st.columns(2)
            with col1:
                fe_column = st.selectbox("元になる列", ["(なし)"] + df_cols)
            with col2:
                fe_method = st.selectbox("処理方法", [
                    "何もしない", "文字数カウント", "数値抽出", 
                    "キーワード有無フラグ", "ダミー変数化 (One-Hot)", "数値のビン分割 (10分割)"
                ])
            col3, col4 = st.columns(2)
            with col3:
                fe_prefix = st.text_input("特徴量の新しい列名の接頭辞")
            with col4:
                fe_keyword = st.text_input("キーワード（有無フラグ用）")

        with st.expander("🗑 フィルタ・列削除"):
            col1, col2, col3 = st.columns(3)
            with col1:
                range_column = st.selectbox("数値フィルタ対象列", ["(なし)"] + df_cols)
            with col2:
                min_value = st.text_input("最小値")
            with col3:
                max_value = st.text_input("最大値")
            drop_columns_text = st.text_input("削除する列名（カンマ区切り）")

        # 前処理実行ボタン
        if st.button("前処理を実行", type="primary"):
            with st.spinner("処理中..."):
                df = st.session_state.df.copy()
                log = []
                target = target_col if target_col != "(なし)" else None

                # 1. 基本処理
                if clean_text:
                    text_cols = df.select_dtypes(include=["object", "string"]).columns
                    for col in text_cols:
                        df[col] = df[col].map(clean_text_value)
                    log.append(f"文字列整理：{len(text_cols)}列の空白・改行を整理")

                if missing_action == "数値列を中央値で補完":
                    for col in get_numeric_columns(df):
                        if col != target and df[col].isna().any():
                            df[col] = df[col].fillna(df[col].median())
                    log.append("欠損値補完：数値列を中央値で補完（目的変数は除く）")
                elif missing_action == "欠損行を削除":
                    df = df.dropna().copy()
                    log.append("欠損行削除：欠損を含む行をすべて削除")

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
                    log.append("異常値削除：IQR基準で外れ値を含む行を削除")

                # 2. フィルタ・列削除
                if range_column != "(なし)":
                    numeric = pd.to_numeric(df[range_column], errors="coerce")
                    mask = pd.Series(True, index=df.index)
                    if min_value.strip(): mask &= numeric >= float(min_value)
                    if max_value.strip(): mask &= numeric <= float(max_value)
                    df = df.loc[mask].copy()
                    log.append(f"範囲フィルタ：{range_column} の指定範囲外を削除")

                if drop_columns_text.strip():
                    to_drop = [x.strip() for x in drop_columns_text.split(",") if x.strip() in df.columns and x.strip() != target]
                    if to_drop:
                        df = df.drop(columns=to_drop)
                        log.append(f"列削除：{', '.join(to_drop)} を削除")

                # 3. 文字列加工・分割
                if replace_column != "(なし)" and remove_text.strip():
                    df[replace_column] = df[replace_column].astype(str).str.replace(remove_text, "", regex=False)
                    log.append(f"文字列削除：{replace_column} から '{remove_text}' を除去")

                if split_column != "(なし)" and split_delimiter:
                    prefix = split_prefix.strip() or f"{split_column}_分割"
                    n_parts = int(split_max_parts)
                    split_values = df[split_column].astype(str).str.split(re.escape(split_delimiter), n=n_parts - 1, expand=True)
                    for i in range(split_values.shape[1]):
                        df[f"{prefix}_{i+1}"] = split_values.iloc[:, i].str.strip()
                    log.append(f"カラム分割：{split_column} を '{split_delimiter}' で最大 {n_parts} 分割")

                # 4. 特徴量エンジニアリング
                if fe_column != "(なし)" and fe_method != "何もしない":
                    source = df[fe_column]
                    prefix = fe_prefix.strip() or f"{fe_column}_特徴"
                    
                    if fe_method == "文字数カウント":
                        df[f"{prefix}_文字数"] = source.astype(str).str.len()
                        log.append(f"特徴量追加：{fe_column} の文字数をカウント")
                        
                    elif fe_method == "数値抽出":
                        extracted = source.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
                        df[f"{prefix}_数値"] = pd.to_numeric(extracted, errors="coerce")
                        log.append(f"特徴量追加：{fe_column} から数値を抽出")
                        
                    elif fe_method == "キーワード有無フラグ":
                        if fe_keyword.strip():
                            df[f"{prefix}_{fe_keyword}_有無"] = source.astype(str).str.contains(re.escape(fe_keyword)).astype(int)
                            log.append(f"特徴量追加：{fe_column} に '{fe_keyword}' が含まれるかフラグ化")
                            
                    elif fe_method == "ダミー変数化 (One-Hot)":
                        dummies = pd.get_dummies(source, prefix=prefix, dummy_na=False).astype(int)
                        df = pd.concat([df, dummies], axis=1)
                        log.append(f"特徴量追加：{fe_column} をダミー変数化")
                        
                    elif fe_method == "数値のビン分割 (10分割)":
                        num_series = pd.to_numeric(source, errors="coerce")
                        df[f"{prefix}_ビン"] = pd.qcut(num_series, q=10, labels=False, duplicates='drop')
                        log.append(f"特徴量追加：{fe_column} を10段階のビンに分割")

                st.session_state.processed_df = df
                st.session_state.logs = log
                st.success("前処理が完了しました！「3. 結果確認・ダウンロード」タブを開いてください。")
    else:
        st.warning("「1. データ読み込み・確認」タブでCSVをアップロードしてください。")

with tab3:
    if st.session_state.processed_df is not None:
        st.subheader("処理ログ")
        for log in st.session_state.logs:
            st.write(f"✅ {log}")
            
        st.subheader(f"加工後データ (全 {st.session_state.processed_df.shape[0]}行 × {st.session_state.processed_df.shape[1]}列)")
        st.dataframe(st.session_state.processed_df.head(100), use_container_width=True)
        
        # CSVダウンロード用データの作成
        csv = st.session_state.processed_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 加工済みCSVをダウンロード",
            data=csv,
            file_name="preprocessed_data.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("前処理が完了すると、ここに結果とダウンロードボタンが表示されます。")
