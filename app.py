import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.subplots as plt
import seaborn as sns
import re
import io
import zipfile

# グラフの日本語文字化け対策（インストールされている場合のみ適用）
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
        for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc-jp"]:
            try:
                if hasattr(file_bytes, 'seek'):
                    file_bytes.seek(0)
                return pd.read_csv(file_bytes, encoding=enc, low_memory=False)
            except Exception as e:
                errors.append(f"CSV({enc}): {e}")
                
        # エラー無視モードで強制突破を試みる
        for enc in ["cp932", "utf-8"]:
            try:
                if hasattr(file_bytes, 'seek'):
                    file_bytes.seek(0)
                return pd.read_csv(file_bytes, encoding=enc, errors='replace', low_memory=False)
            except Exception:
                pass
                
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

import matplotlib.pyplot as plt
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

    if st.session_state.df is not None:
        df_cols = [c for c in st.session_state.df.columns.tolist() if c != "_source_"]
        default_target = next((c for c in TARGET_CANDIDATES if c in df_cols), df_cols[0])
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("---")
        st.subheader("🛠 前処理ワークフロー（STEP1 〜 4）")
        target_col = st.selectbox("🎯 目的変数（削除や補完から除外して守る列）", ["(なし)"] + df_cols, index=df_cols.index(default_target)+1 if default_target in df_cols else 0)

        # ----------------------------------------------------------------
        # STEP1: 文字列の処理 (分割 / ダミー化)
        # ----------------------------------------------------------------
        with st.expander("🪄 STEP1: 複雑な文字列の処理 (分割 / ダミー化) 🌟", expanded=True):
            st.info("💡 LightGBMなどを使うために「文字列のまま分割して残す」か、線形モデル等のために「0と1のフラグに変換する」かを選べます。")
            
            tab1, tab2 = st.tabs(["✂️ 文字列のまま複数列に分割 (LightGBM向け)", "🔄 0/1のフラグ列に変換 (ダミー化)"])
            
            with tab1:
                st.markdown("##### ✂️ 記号や空白で文字をバラバラにして列を分ける")
                c1, c2 = st.columns([5, 5])
                with c1:
                    split_col_step1 = st.selectbox("分割する列", ["(なし)"] + df_cols, key="split_col_step1")
                    split_strategy = st.radio("分割の方式", ["単語ごとに専用列を作成 (ズレ完全防止・推奨)", "左詰めで分割 (最大N列指定)"])
                    if split_strategy == "左詰めで分割 (最大N列指定)":
                        split_max = st.number_input("最大何列に分けるか", min_value=2, max_value=50, value=5)
                        split_sort_mode = st.selectbox("行内の単語のソート方法", ["五十音・アルファベット順", "全体の出現頻度順", "ソートしない"])
                with c2:
                    split_delim = st.text_input("区切り文字", value="自動認識 (/, |, [], 空白, 全角半角)")
                    split_drop_step1 = st.checkbox("処理後に元の列を削除する", value=True, key="split_drop_step1")
            
            with tab2:
                st.markdown("##### 🔄 複数キーワードの一括フラグ化")
                c1, c2 = st.columns([6, 4])
                with c1:
                    ml_column = st.selectbox("複雑な文字列が含まれる列を選ぶ", ["(なし)"] + df_cols, key="ml_col")
                with c2:
                    ml_delim = st.text_input("区切り文字 (「／」や「空白」)", value="／", key="ml_delim")
                ml_drop_original = st.checkbox("処理後に元の列を自動で削除する", value=True, key="ml_drop")
                
                st.divider()
                st.markdown("##### 🔄 通常のカテゴリダミー化 (One-Hot)")
                onehot_columns = st.multiselect("ダミー化（0と1に変換）する列を複数選択", df_cols)

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
                replace_column = st.selectbox("一部の文字を消す列", ["(なし)"] + df_cols, key="replace_col")
                extract_column = st.selectbox("特定の文字だけ残す(抽出)列", ["(なし)"] + df_cols, key="extract_col")
            with c2:
                remove_text = st.text_input("消したい文字 (カンマ区切り可)")
                extract_text = st.text_input("残したい文字 (カンマ区切り可)")

        # ----------------------------------------------------------------
        # STEP3: その他の特徴量作成
        # ----------------------------------------------------------------
        with st.expander("✨ STEP3: その他の特徴量作成 (時系列・数値・集約など)", expanded=True):
            c1, c2 = st.columns([4, 6])
            with c1:
                fe_column = st.selectbox("基準となる列（元データ）", ["(なし)"] + df_cols)
            with c2:
                fe_method = st.selectbox("処理方法を選択", [
                    "何もしない", 
                    "【集約】カテゴリごとの平均値などを算出 (複数選択可)",
                    "【変換】カテゴリを頻度（出現回数）に変換 (Count Encoding)", 
                    "【変換】カテゴリを数値IDに変換 (Label Encoding - 決定木向け)",
                    "【型変換】カテゴリ型 (category) に変換 (LightGBMネイティブ用)",
                    "【変換】数値を任意の幅でグループ化 (例: 10刻みで「〇〇代」)",
                    "【数値計算】2つの列で四則演算 (+, -, *, /)",
                    "【数値計算】対数変換 (Log1p - 歪み補正)",
                    "【変換】数値のビン分割 (10分割)",
                    "【時系列】日付列から年・月・日・曜日を抽出",  
                    "【時系列】日付のフォーマット変換", 
                    "【抽出】「徒歩〇分」から最短分数を抽出", 
                    "【抽出】「築〇年」から年数を抽出 (単一の数値に変換)", 
                    "【抽出】「〇年〇ヶ月」を年と月に別々の列に分割",
                    "【抽出】最初の数値を抽出", 
                    "【テキスト】文字数カウント", 
                    "【フラグ】キーワード有無フラグ"
                ])
            
            c3, c4 = st.columns([4, 6])
            with c3:
                fe_prefix = st.text_input("追加する特徴量の接頭辞 (任意)", placeholder="例: feat")
                
            with c4:
                date_format = None
                fe_keyword = ""
                fe_col2 = None
                fe_op = "+"
                fe_num_col = None
                fe_aggs = ["mean"]
                fe_group_width = 10
                fe_group_suffix = "代"

                if fe_method == "【時系列】日付のフォーマット変換":
                    date_format = st.selectbox("変換後のフォーマット", [
                        "yyyy-m-d (例: 2014-1-5) ※ゼロ埋めなし",
                        "yyyy-MM-dd (例: 2014-01-05)",
                        "yyyy/m/d (例: 2014/1/5)",
                        "yyyy/MM/dd (例: 2014/01/05)",
                        "yyyyMMdd (例: 20140105)"
                    ])
                elif fe_method == "【型変換】カテゴリ型 (category) に変換 (LightGBMネイティブ用)":
                    st.info("💡 category型を保存するには、ダウンロード時に「Parquet形式」を選んでください。（CSV形式だと文字列に戻ってしまいます）")
                elif fe_method == "【変換】数値を任意の幅でグループ化 (例: 10刻みで「〇〇代」)":
                    col4_1, col4_2 = st.columns(2)
                    with col4_1:
                        fe_group_width = st.number_input("区切り幅 (例: 10, 5, 100)", min_value=1, value=10)
                    with col4_2:
                        fe_group_suffix = st.text_input("後ろにつける文字 (例: 代, 度台, 円台)", value="代")
                elif fe_method == "【フラグ】キーワード有無フラグ":
                    fe_keyword = st.text_input("キーワード (カンマ区切りで複数可)", placeholder="例: カレー, ハンバーグ, 魚")
                elif fe_method == "【数値計算】2つの列で四則演算 (+, -, *, /)":
                    col4_1, col4_2 = st.columns([3, 7])
                    with col4_1:
                        fe_op = st.radio("計算記号", ["+", "-", "×", "÷"], horizontal=True)
                    with col4_2:
                        fe_col2 = st.selectbox("計算する2つ目の列", df_cols)
                elif fe_method == "【集約】カテゴリごとの平均値などを算出 (複数選択可)":
                    col4_1, col4_2 = st.columns(2)
                    with col4_1:
                        fe_num_col = st.selectbox("集計する数値列 (例: y, 売上)", ["(なし)"] + get_numeric_columns(st.session_state.df))
                    with col4_2:
                        fe_aggs = st.multiselect("集計方法 (複数選択可)", ["mean", "median", "sum", "std", "max", "min", "count"], default=["mean"])

        # ----------------------------------------------------------------
        # STEP4: 大掃除
        # ----------------------------------------------------------------
        with st.expander("🗑 STEP4: 大掃除 (いらない列の削除)", expanded=True):
            st.markdown("##### 🧹 カラム名（列の名前）を指定して一括削除")
            drop_by_name = st.text_input("列名にこの文字が含まれていたらすべて削除 (カンマ区切り可)", help="例: 脱衣所, 洗面台独立")
            
            st.divider()
            st.markdown("##### 📉 マイナーなフラグ列を自動削除")
            min_count = st.number_input("出現件数がこれ以下のダミー列を自動削除", min_value=0, value=0)

            st.divider()
            st.markdown("##### 🗑 手提で選んで削除")
            drop_columns = st.multiselect("リストから選択", df_cols)

            st.divider()
            st.markdown("##### 🎯 数値の範囲でフィルタ (行の削除)")
            c1, c2 = st.columns(2)
            with c1: 
                range_column = st.selectbox("数値フィルタ列 (指定範囲外を削除)", ["(なし)"] + df_cols)
            with c2: 
                min_value = st.text_input("残す最小値")
                max_value = st.text_input("残す最大値")

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
                # STEP1: 分割 ＆ フラグ化
                # -------------------------
                if split_col_step1 != "(なし)":
                    prefix = f"{split_col_step1}_分割"
                    temp_series = df[split_col_step1].fillna("").astype(str)
                    
                    if split_delim == "自動認識 (/, |, [], 空白, 全角半角)":
                        temp_series = temp_series.str.replace(r'[／/｜|\[\]［］\s]+', '|||', regex=True).str.strip('|||')
                        actual_delim = '|||'
                    elif split_delim == "空白":
                        temp_series = temp_series.str.replace(r'\s+', '|||', regex=True).str.strip('|||')
                        actual_delim = '|||'
                    else:
                        actual_delim = split_delim

                    if split_strategy == "単語ごとに専用列を作成 (ズレ完全防止・推奨)":
                        dummies = temp_series.str.get_dummies(sep=actual_delim)
                        if "" in dummies.columns:
                            dummies = dummies.drop(columns=[""])
                        if "nan" in dummies.columns:
                            dummies = dummies.drop(columns=["nan"])
                            
                        for c in dummies.columns:
                            df[f"{prefix}_{c}"] = dummies[c].map({1: c, 0: np.nan})
                        log.append(f"✂️ カラム分割：{split_col_step1} を単語ごとの専用列（{len(dummies.columns)}列）に分割し、ズレを完全に解消しました")
                    
                    else:
                        n_parts = int(split_max)
                        def split_and_align(text):
                            if pd.isna(text) or text == "": return []
                            return [p.strip() for p in str(text).split(actual_delim) if p.strip()]
                            
                        split_lists = temp_series.apply(split_and_align)
                        
                        if split_sort_mode != "ソートしない":
                            all_words = []
                            for lst in split_lists: all_words.extend(lst)
                            word_series = pd.Series(all_words)
                            if split_sort_mode == "五十音・アルファベット順":
                                sorted_unique = sorted(list(word_series.unique()))
                                order_map = {w: i for i, w in enumerate(sorted_unique)}
                            else:
                                freq_ranks = word_series.value_counts()
                                order_map = {w: i for i, (w, _) in enumerate(freq_ranks.items())}
                            
                            def apply_sort(parts):
                                return sorted(list(set(parts)), key=lambda x: order_map.get(x, 999999))
                            split_lists = split_lists.apply(apply_sort)

                        for i in range(n_parts):
                            df[f"{prefix}_{i+1}"] = split_lists.apply(lambda x: x[i] if i < len(x) else np.nan)
                        log.append(f"✂️ カラム分割：{split_col_step1} を文字列のまま {n_parts} 列に左詰めで分割")
                    
                    if split_drop_step1 and split_col_step1 in df.columns:
                        df = df.drop(columns=[split_col_step1])

                if ml_column != "(なし)":
                    delim = ml_delim if ml_delim else "/"
                    temp_series = df[ml_column].fillna("").astype(str)
                    
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
                        
                    prefix = ml_column
                    dummies.columns = [f"{prefix}_{c}" for c in dummies.columns]
                    df = pd.concat([df, dummies], axis=1)
                    log.append(f"🪄 一括フラグ化：{ml_column} を {len(dummies.columns)}列のダミー変数に変換")
                    
                    if ml_drop_original and ml_column in df.columns:
                        df = df.drop(columns=[ml_column])

                if onehot_columns:
                    df = pd.get_dummies(df, columns=onehot_columns, dummy_na=False, dtype=int)
                    log.append(f"🔄 One-Hot変換：{', '.join(onehot_columns)} を変換")

                # -------------------------
                # STEP2: 基本処理
                # -------------------------
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

                # -------------------------
                # STEP3: その他の特徴量作成
                # -------------------------
                if fe_column != "(なし)" and fe_method != "何もしない":
                    source = df[fe_column]
                    prefix = fe_prefix.strip() or f"{fe_column}"
                    
                    if fe_method == "【変換】カテゴリを数値IDに変換 (Label Encoding - 決定木向け)":
                        codes, uniques = pd.factorize(source)
                        df[f"{prefix}_ID"] = np.where(pd.isna(source), np.nan, codes)
                        log.append(f"特徴量追加：{fe_column} を数値ID（Label Encoding）に変換")

                    elif fe_method == "【変換】カテゴリを頻度（出現回数）に変換 (Count Encoding)":
                        freq_map = source.value_counts(dropna=False)
                        df[f"{prefix}_頻度"] = source.map(freq_map)
                        log.append(f"特徴量追加：{fe_column} を頻度(Count Encoding)に変換")
                        
                    elif fe_method == "【型変換】カテゴリ型 (category) に変換 (LightGBMネイティブ用)":
                        df[f"{prefix}_cat"] = source.astype("category")
                        log.append(f"型変換：{fe_column} を category 型に変換")

                    elif fe_method == "【変換】数値を任意の幅でグループ化 (例: 10刻みで「〇〇代」)":
                        def to_group(val):
                            try:
                                num = float(val)
                                if pd.isna(num): return np.nan
                                return f"{int(np.floor(num / fe_group_width) * fe_group_width)}{fe_group_suffix}"
                            except Exception:
                                return np.nan
                        df[f"{prefix}_{fe_group_width}区切り"] = source.apply(to_group)
                        log.append(f"特徴量追加：{fe_column} を {fe_group_width} 刻みのカテゴリに変換")

                    elif fe_method == "【時系列】日付列から年・月・日・曜日を抽出":
                        dt_series = pd.to_datetime(source, errors="coerce")
                        df[f"{prefix}_年"] = dt_series.dt.year
                        df[f"{prefix}_月"] = dt_series.dt.month
                        df[f"{prefix}_日"] = dt_series.dt.day
                        df[f"{prefix}_曜日"] = dt_series.dt.dayofweek
                        log.append(f"特徴量追加：{fe_column} から「年」「月」「日」「曜日」をそれぞれ抽出")
                        
                    elif fe_method == "【時系列】日付のフォーマット変換":
                        dt_series = pd.to_datetime(source, errors="coerce")
                        def format_date(x, fmt):
                            if pd.isna(x): return np.nan
                            if "yyyy-m-d" in fmt:
                                return f"{x.year}-{x.month}-{x.day}"
                            elif "yyyy-MM-dd" in fmt:
                                return x.strftime("%Y-%m-%d")
                            elif "yyyy/m/d" in fmt:
                                return f"{x.year}/{x.month}/{x.day}"
                            elif "yyyy/MM/dd" in fmt:
                                return x.strftime("%Y/%m/%d")
                            elif "yyyyMMdd" in fmt:
                                return x.strftime("%Y%m%d")
                            return np.nan
                            
                        df[f"{prefix}_フォーマット済"] = dt_series.apply(lambda x: format_date(x, date_format))
                        log.append(f"フォーマット変換：{fe_column} を「{date_format.split(' ')[0]}」形式に変換")

                    elif fe_method == "【数値計算】2つの列で四則演算 (+, -, *, /)":
                        if fe_col2 and fe_col2 != "(なし)":
                            s1 = pd.to_numeric(df[fe_column], errors="coerce")
                            s2 = pd.to_numeric(df[fe_col2], errors="coerce")
                            op_name = {"+": "和", "-": "差", "×": "積", "÷": "商"}[fe_op]
                            if fe_op == "+": res = s1 + s2
                            elif fe_op == "-": res = s1 - s2
                            elif fe_op == "×": res = s1 * s2
                            elif fe_op == "÷": res = np.where(s2 == 0, np.nan, s1 / s2)
                            df[f"{prefix}_{op_name}_{fe_col2}"] = res
                            log.append(f"特徴量追加：{fe_column} と {fe_col2} の四則演算({fe_op})")

                    elif fe_method == "【集約】カテゴリごとの平均値などを算出 (複数選択可)":
                        if fe_num_col and fe_num_col != "(なし)" and len(fe_aggs) > 0:
                            num_series = pd.to_numeric(df[fe_num_col], errors="coerce")
                            temp_df = pd.DataFrame({"cat": df[fe_column], "num": num_series, "src": df.get("_source_", "single")})
                            
                            for agg_func in fe_aggs:
                                if "train" in temp_df["src"].values:
                                    agg_map = temp_df[temp_df["src"] == "train"].groupby("cat")["num"].agg(agg_func)
                                else:
                                    agg_map = temp_df.groupby("cat")["num"].agg(agg_func)
                                    
                                df[f"{prefix}_{fe_num_col}_{agg_func}"] = df[fe_column].map(agg_map)
                            log.append(f"特徴量追加：{fe_column} ごとの {fe_num_col} の集約値({', '.join(fe_aggs)})を算出")

                    elif fe_method == "【数値計算】対数変換 (Log1p - 歪み補正)":
                        num_s = pd.to_numeric(source, errors="coerce")
                        df[f"{prefix}_log1p"] = np.log1p(np.clip(num_s, 0, None))
                        log.append(f"特徴量追加：{fe_column} を対数変換(Log1p)")

                    elif fe_method == "【テキスト】文字数カウント":
                        df[f"{prefix}_文字数"] = source.astype(str).str.len()
                        log.append(f"特徴量追加：{fe_column} の文字数")
                        
                    elif fe_method == "【抽出】最初の数値を抽出":
                        extracted = source.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
                        df[f"{prefix}_数値"] = pd.to_numeric(extracted, errors="coerce")
                        log.append(f"特徴量追加：{fe_column} から数値抽出")
                        
                    elif fe_method == "【抽出】「徒歩〇分」から最短分数を抽出":
                        def get_min_walk(text):
                            if pd.isna(text): return np.nan
                            matches = re.findall(r"徒歩\s*(\d+)\s*分?", str(text))
                            if matches: return min(map(int, matches))
                            return np.nan
                        df[f"{prefix}_最短徒歩分"] = source.apply(get_min_walk)
                        log.append(f"特徴量追加：{fe_column} から徒歩最短分数")
                        
                    elif fe_method == "【抽出】「築〇年」から年数を抽出 (単一の数値に変換)":
                        def get_building_age(text):
                            if pd.isna(text): return np.nan
                            s = str(text)
                            if "新築" in s: return 0.0
                            y_match = re.search(r"(\d+)\s*年", s)
                            m_match = re.search(r"(\d+)\s*[ヶ箇カかケヵ]?月", s)
                            y = int(y_match.group(1)) if y_match else 0
                            m = int(m_match.group(1)) if m_match else 0
                            if not y_match and not m_match: return np.nan
                            return y + (m / 12.0)
                        df[f"{prefix}_築年数"] = source.apply(get_building_age)
                        log.append(f"特徴量追加：{fe_column} から築年数抽出(小数点変換)")
                        
                    elif fe_method == "【抽出】「〇年〇ヶ月」を年と月に別々の列に分割":
                        def extract_year(text):
                            if pd.isna(text): return np.nan
                            s = str(text)
                            if "新築" in s: return 0.0
                            match = re.search(r"(\d+)\s*年", s)
                            if match: return float(match.group(1))
                            if re.search(r"(\d+)\s*[ヶ箇カかケヵ]?月", s): return 0.0
                            return np.nan

                        def extract_month(text):
                            if pd.isna(text): return np.nan
                            s = str(text)
                            if "新築" in s: return 0.0
                            match = re.search(r"(\d+)\s*[ヶ箇カかケヵ]?月", s)
                            if match: return float(match.group(1))
                            if re.search(r"(\d+)\s*年", s): return 0.0
                            return np.nan

                        df[f"{prefix}_年"] = source.apply(extract_year)
                        df[f"{prefix}_ヶ月"] = source.apply(extract_month)
                        log.append(f"特徴量追加：{fe_column} を「年」と「ヶ月」の2列に分割")

                    elif fe_method == "【フラグ】キーワード有無フラグ":
                        if fe_keyword.strip():
                            kws = [k.strip() for k in fe_keyword.split(",") if k.strip()]
                            for kw in kws:
                                df[f"{prefix}_{kw}_有無"] = source.astype(str).str.contains(re.escape(kw)).astype(int)
                            log.append(f"特徴量追加：{fe_column} から '{', '.join(kws)}' のフラグを作成 ({len(kws)}列追加)")
                            
                    elif fe_method == "【変換】数値のビン分割 (10分割)":
                        num_series = pd.to_numeric(source, errors="coerce")
                        df[f"{prefix}_ビン"] = pd.qcut(num_series, q=10, labels=False, duplicates='drop')
                        log.append(f"特徴量追加：{fe_column} をビン分割")

                # -------------------------
                # STEP4: 大掃除
                # -------------------------
                if drop_by_name.strip():
                    drop_words = [w.strip() for w in drop_by_name.split(",") if w.strip()]
                    to_drop_by_name = []
                    for col in df.columns:
                        if col in [target, "_source_"]: continue
                        if any(w in col for w in drop_words):
                            to_drop_by_name.append(col)
                    if to_drop_by_name:
                        df = df.drop(columns=to_drop_by_name)
                        log.append(f"🧹 名前で一括削除：名前に '{', '.join(drop_words)}' を含む {len(to_drop_by_name)}列 を削除")

                if min_count > 0:
                    rare_dropped = 0
                    for col in list(df.columns):
                        if col in [target, "_source_"]: continue
                        if pd.api.types.is_numeric_dtype(df[col]):
                            unique_vals = df[col].dropna().unique()
                            if set(unique_vals).issubset({0, 1}):
                                if df[col].sum() <= min_count:
                                    df = df.drop(columns=[col])
                                    rare_dropped += 1
                    if rare_dropped > 0:
                        log.append(f"📉 マイナーフラグ削除：出現回数 {int(min_count)} 以下の列を {rare_dropped}列 自動削除")

                if drop_columns:
                    to_drop = [c for c in drop_columns if c in df.columns and c not in [target, "_source_"]]
                    if to_drop:
                        df = df.drop(columns=to_drop)
                        log.append(f"🗑 手動削除：手動で選んだ {len(to_drop)}列 を削除")

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
        
        view_tabs = st.tabs(["データ確認 ＆ ダウンロード", "プロファイル・分布", "📊 グループ集計 (EDA)", "🔥 相関ヒートマップ"])
        
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
                st.info("👇このボタンを押すと、右上の表に表示されている結果を「元のデータ」として確定させ、さらに不要な列の削除などを重ねていくことができます。（何度でも繰り返せます）")
                if st.button("現在の結果をベースに設定を続ける", use_container_width=True):
                    st.session_state.df = target_df
                    st.session_state.processed_df = None
                    st.session_state.logs = []
                    st.rerun()

            st.divider()
            st.markdown("### 📥 データのダウンロード")
            save_format = st.radio("「**保存形式を選択**」", ["通常のCSV (.csv)", "GZIP圧縮CSV (.csv.gz)", "Parquet形式 (.parquet)"], horizontal=True)
            
            no_header = st.checkbox("CSVのヘッダー（列名）を含めずにダウンロードする（※コンペの応募ファイル用）", value=False)
            export_header = not no_header
            
            source_vals = target_df.get("_source_", pd.Series()).unique()
            
            if "train" in source_vals and "test" in source_vals:
                train_out = target_df[target_df["_source_"] == "train"].drop(columns=["_source_"], errors="ignore")
                test_out = target_df[target_df["_source_"] == "test"].drop(columns=["_source_"], errors="ignore")
                
                train_data, ext, mime = convert_df(train_out, save_format, header=export_header)
                test_data, _, _ = convert_df(test_out, save_format, header=export_header)
                
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
                out_data, ext, mime = convert_df(target_df.drop(columns=["_source_"], errors="ignore"), save_format, header=export_header)
                reduction = (1 - len(out_data) / st.session_state.original_size) * 100 if st.session_state.original_size > 0 else 0
                
                st.write(f"処理後サイズ: {len(out_data) / (1024*1024):.2f} MB (削減率: {reduction:.1f}%)")
                st.download_button("「**ダウンロード**」", out_data, f"preprocessed_data.{ext}", mime, type="primary")

        with view_tabs[1]:
            st.markdown("**全カラムの基本情報**")
            st.dataframe(profile_table(target_df), use_container_width=True)
            st.divider()
            st.markdown("**ヒストグラム**")
            if num_cols_list:
                hist_col = st.selectbox("分布を確認する列", num_cols_list)
                fig = make_histogram(target_df, hist_col)
                if fig:
                    st.pyplot(fig)
            else:
                st.info("数値列がありません。")
                
        with view_tabs[2]:
            st.markdown("### 🔍 カテゴリごとの統計情報を確認")
            st.write("曜日や天気などのカテゴリごとに、販売数などの数値（count, mean, median, sum, std）を一発で集計・比較します。")
            
            df_cols_clean = [c for c in target_df.columns if c != "_source_"]
            cat_candidates = [c for c in df_cols_clean if c not in num_cols_list]
            if not cat_candidates:
                cat_candidates = df_cols_clean
                
            c_eda1, c_eda2 = st.columns(2)
            with c_eda1:
                grp_col = st.selectbox("グループ化する列 (曜日など)", ["(なし)"] + df_cols_clean)
            with c_eda2:
                tgt_col = st.selectbox("集計する数値列 (販売数など)", ["(なし)"] + num_cols_list)
                
            if grp_col != "(なし)" and tgt_col != "(なし)":
                try:
                    s_tgt = pd.to_numeric(target_df[tgt_col], errors="coerce")
                    agg_df = s_tgt.groupby(target_df[grp_col]).agg(["count", "sum", "mean", "median", "std"]).round(1)
                    
                    weekday_order = ["月", "火", "水", "木", "金", "土", "日"]
                    plot_order = None
                    if any(w in agg_df.index for w in weekday_order):
                        sorted_idx = [d for d in weekday_order if d in agg_df.index]
                        other_idx = [d for d in agg_df.index if d not in weekday_order]
                        agg_df = agg_df.reindex(sorted_idx + other_idx)
                        plot_order = sorted_idx + other_idx
                        st.info("💡 曜日データを検知したため、月曜日〜日曜日の順に自動ソートしました。")
                        
                    st.dataframe(agg_df, use_container_width=True)
                    
                    # ★今回追加：箱ひげ図の描画
                    st.markdown("#### 📦 箱ひげ図 (分布と外れ値の確認)")
                    fig_box, ax_box = plt.subplots(figsize=(10, 6))
                    
                    # 欠損値や数値化できないデータを除外して描画用にクリーンアップ
                    plot_df = target_df[[grp_col, tgt_col]].dropna()
                    plot_df[tgt_col] = pd.to_numeric(plot_df[tgt_col], errors="coerce")
                    plot_df = plot_df.dropna()
                    
                    # 箱ひげ図を描画（さらにストリッププロットで実データを薄く重ねる）
                    sns.boxplot(data=plot_df, x=grp_col, y=tgt_col, order=plot_order, ax=ax_box, palette="Set3", showfliers=False)
                    sns.stripplot(data=plot_df, x=grp_col, y=tgt_col, order=plot_order, ax=ax_box, color='black', alpha=0.3, size=4, jitter=True)
                    
                    ax_box.set_title(f"{grp_col} ごとの {tgt_col} の分布", fontsize=12)
                    ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45, ha='right')
                    fig_box.tight_layout()
                    st.pyplot(fig_box)
                    
                except Exception as e:
                    st.warning(f"集計時にエラーが発生しました: {e}")

        with view_tabs[3]:
            st.markdown("### 🔥 変数間の相関関係 (ヒートマップ)")
            st.write("数値データ同士の関連性の強さ（相関係数: -1.0 〜 1.0）を確認します。")
            
            if len(num_cols_list) >= 2:
                st.write("※列数が多すぎると文字が潰れるため、表示する列を絞り込むことができます。")
                corr_cols = st.multiselect("相関を確認する列を選択", num_cols_list, default=num_cols_list[:min(15, len(num_cols_list))])
                
                if len(corr_cols) >= 2:
                    corr_matrix = target_df[corr_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                                vmin=-1, vmax=1, square=True, ax=ax, 
                                cbar_kws={"shrink": .8}, annot_kws={"size": 9})
                    
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                    fig.tight_layout()
                    
                    st.pyplot(fig)
                    
                    if target_col != "(なし)" and target_col in corr_cols:
                        st.markdown(f"**🎯 目的変数「{target_col}」との相関ランキング**")
                        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
                        st.dataframe(target_corr.to_frame("相関係数"), use_container_width=True)
                else:
                    st.warning("相関を確認するには、2つ以上の列を選択してください。")
            else:
                st.info("数値列が2つ以上ないため、相関行列を計算できません。")
