import streamlit as st
import pandas as pd
import numpy as np

# --- ページとCSSの設定 ---
st.set_page_config(page_title="Kaggle前処理捜査ファイル", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    .reportview-container { background-color: #1E1E2E; color: white; }
    .sidebar .sidebar-content { background-color: #2D2D3D; }
    div.stButton > button:first-child {
        background-color: #3B3B52; color: #E0E0E0; border: 1px solid #555; border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- ★データ軽量化（ダウンキャスト）関数の修正版 ---
def reduce_mem_usage(df):
    """数値データ（int, float）のみを対象に型を最小化してメモリを削減する"""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        
        # エラー対策：データ型がintかfloatで始まるものだけを処理する
        if str(col_type)[:3] == 'int' or str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    return df, start_mem, end_mem

# --- セッションステートの初期化 ---
if "df" not in st.session_state:
    st.session_state.df = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "merge_success" not in st.session_state:
    st.session_state.merge_success = False
if "output_name" not in st.session_state:
    st.session_state.output_name = "data.csv"

st.title("Kaggle前処理捜査ファイル")
st.write("CSVを読み込み → 基本統計・欠損・分布・相関を確認 (端末内で処理、送信なし)")

# --- ファイルアップロード領域 ---
st.markdown("### CSVを選択 / タップしてアップロード")
uploaded_files = st.file_uploader("複数ファイル可・大容量はストリーミング読込", type=["csv"], accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    max_rows = st.number_input("最大行数/ファイル", value=1000000, step=100000)
with col2:
    sampling = st.selectbox("間引き", ["全行", "10%", "1%"])

if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    selected_file_name = st.radio("ベースとなるファイルを選択:", file_names, horizontal=True)
    
    if st.session_state.current_file != selected_file_name:
        selected_file = next(f for f in uploaded_files if f.name == selected_file_name)
        
        if sampling == "10%":
            temp_df = pd.read_csv(selected_file, nrows=max_rows)
            temp_df = temp_df.sample(frac=0.1, random_state=42)
        elif sampling == "1%":
            temp_df = pd.read_csv(selected_file, nrows=max_rows)
            temp_df = temp_df.sample(frac=0.01, random_state=42)
        else:
            temp_df = pd.read_csv(selected_file, nrows=max_rows)
            
        # 軽量化処理を実行
        optimized_df, start_mem, end_mem = reduce_mem_usage(temp_df)
        st.toast(f"メモリ使用量を {start_mem:.2f} MB から {end_mem:.2f} MB に最適化しました！")
        
        st.session_state.df = optimized_df
        st.session_state.current_file = selected_file_name
        st.session_state.merge_success = False
        st.session_state.output_name = selected_file_name
        st.rerun()

    df = st.session_state.df
    st.write(f"現在のデータ ({st.session_state.output_name}): {df.shape[0]:,}行 × {df.shape[1]}列")

    # --- 分析・処理タブ ---
    tabs = st.tabs([
        "先頭5行", "基本統計", "欠損", "ヒストグラム", "文字列カウント", 
        "相関係数", "時系列", "クリーニング", "エンコード", "結合", "CSV保存"
    ])
    
    with tabs[0]:
        st.write(f"先頭5行 (全{df.shape[0]:,}行)")
        st.dataframe(df.head(), use_container_width=True)
        
    with tabs[1]:
        st.write("基本統計量")
        st.dataframe(df.describe(include='all').T, use_container_width=True)
        
    with tabs[2]:
        st.write("列ごとの欠損値数")
        missing = df.isnull().sum().reset_index()
        missing.columns = ['列名', '欠損数']
        missing = missing[missing['欠損数'] > 0].sort_values('欠損数', ascending=False)
        if missing.empty:
            st.success("欠損値はありません。")
        else:
            st.dataframe(missing, use_container_width=True)
            
    with tabs[3]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("列を選択 (ヒストグラム)", numeric_cols)
            st.bar_chart(df[selected_col].value_counts(bins=20, sort=False))
        else:
            st.warning("数値列がありません。")

    with tabs[4]:
        st.markdown("### 文字列（カテゴリ）のカウント")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            selected_cat_col = st.selectbox("集計する列を選択してください", cat_cols)
            count_df = df[selected_cat_col].value_counts().reset_index()
            count_df.columns = [selected_cat_col, '出現回数']
            st.write(f"（**{selected_cat_col}**）の種類: {len(count_df)}件")
            st.dataframe(count_df, use_container_width=True)
        else:
            st.info("文字列（カテゴリ）の列が見つかりません。")

    with tabs[5]:
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.dataframe(numeric_df.corr().style.background_gradient(cmap='coolwarm'), use_container_width=True)
        else:
            st.warning("数値列がありません。")

    with tabs[6]:
        st.markdown("### 時間特徴量の自動抽出")
        date_cols = df.select_dtypes(include=['object', 'datetime']).columns.tolist()
        if date_cols:
            target_date_col = st.selectbox("日付が入力されている列を選択", date_cols)
            if st.button("時間特徴量を生成"):
                try:
                    df_time = df.copy()
                    df_time[target_date_col] = pd.to_datetime(df_time[target_date_col])
                    df_time['year'] = df_time[target_date_col].dt.year
                    df_time['month'] = df_time[target_date_col].dt.month
                    df_time['day'] = df_time[target_date_col].dt.day
                    df_time['dayofweek'] = df_time[target_date_col].dt.dayofweek
                    
                    st.session_state.df = df_time
                    st.success(f"（**{target_date_col}**）から 年・月・日・曜日 を抽出し、新しい列を追加しました！")
                    st.rerun()
                except Exception as e:
                    st.error(f"日付への変換に失敗しました。詳細: {e}")
        else:
            st.info("日付として処理できる列が見つかりません。")

    with tabs[7]:
        st.markdown("### データのクリーニング")
        clean_option = st.radio("操作を選択:", ["列の削除", "欠損値の補完", "欠損行の削除"], horizontal=True)

        if clean_option == "列の削除":
            cols_to_drop = st.multiselect("削除する不要な列を選択してください", df.columns.tolist())
            if st.button("列を削除"):
                if cols_to_drop:
                    st.session_state.df = df.drop(columns=cols_to_drop)
                    st.success(f"（**{', '.join(cols_to_drop)}**）を削除しました！")
                    st.rerun()
                else:
                    st.warning("列が選択されていません。")
        
        elif clean_option == "欠損値の補完":
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                fill_col = st.selectbox("補完する列を選択", missing_cols)
                fill_method = st.selectbox("補完方法", ["0で埋める", "平均値", "中央値", "最頻値", "前の値で埋める(ffill)"])
                
                if st.button("補完を実行"):
                    try:
                        df_clean = df.copy()
                        if fill_method == "0で埋める":
                            df_clean[fill_col] = df_clean[fill_col].fillna(0)
                        elif fill_method == "平均値":
                            df_clean[fill_col] = df_clean[fill_col].fillna(df_clean[fill_col].mean())
                        elif fill_method == "中央値":
                            df_clean[fill_col] = df_clean[fill_col].fillna(df_clean[fill_col].median())
                        elif fill_method == "最頻値":
                            df_clean[fill_col] = df_clean[fill_col].fillna(df_clean[fill_col].mode()[0])
                        elif fill_method == "前の値で埋める(ffill)":
                            df_clean[fill_col] = df_clean[fill_col].fillna(method='ffill')
                            
                        st.session_state.df = df_clean
                        st.success(f"（**{fill_col}**）の欠損値を（**{fill_method}**）で補完しました！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"補完に失敗しました。詳細: {e}")
            else:
                st.info("現在、欠損値のある列はありません。")
        
        elif clean_option == "欠損行の削除":
            missing_rows = df.isnull().any(axis=1).sum()
            st.write(f"現在の欠損を含む行数: （**{missing_rows:,}**）行")
            if missing_rows > 0:
                if st.button("欠損行をすべて削除"):
                    st.session_state.df = df.dropna()
                    st.success("欠損値を含む行をすべて削除しました！")
                    st.rerun()
            else:
                st.info("削除する欠損行はありません。")

    with tabs[8]:
        st.markdown("### ワンホットエンコーディング")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            target_cols = st.multiselect("エンコードする列を選択してください", cat_cols)
            if st.button("ワンホットエンコードを実行"):
                if target_cols:
                    df_encoded = pd.get_dummies(df, columns=target_cols, drop_first=False)
                    st.session_state.df = df_encoded 
                    st.success(f"（**{', '.join(target_cols)}**）をワンホットエンコーディングしました！")
                    st.rerun()
                else:
                    st.warning("列が選択されていません。")
        else:
            st.info("エンコード可能なカテゴリ変数の列が見つかりません。")

    with tabs[9]:
        st.markdown("### データのマージ (キー結合)")
        available_files = [f for f in uploaded_files if f.name != selected_file_name]
        
        if available_files:
            merge_target_name = st.selectbox("結合する追加CSVファイルを選択してください", [f.name for f in available_files])
            
            if merge_target_name:
                target_file = next(f for f in available_files if f.name == merge_target_name)
                # 追加のファイルも読み込み時に軽量化
                df_target, _, _ = reduce_mem_usage(pd.read_csv(target_file))
                
                col1_m, col2_m, col3_m = st.columns(3)
                with col1_m:
                    left_key = st.selectbox("ベースデータのキー列", df.columns.tolist())
                with col2_m:
                    right_key = st.selectbox(f"{merge_target_name}のキー列", df_target.columns.tolist())
                with col3_m:
                    how = st.selectbox("結合方法", ["left", "inner", "outer", "right"])
                
                left_type = df[left_key].dtype
                right_type = df_target[right_key].dtype
                
                if left_type != right_type:
                    st.warning(f"⚠️ キーのデータ型が異なります (ベース: `{left_type}`, 追加データ: `{right_type}`)")
                    force_str = st.checkbox("キーの型を強制的に「文字列」に統一して結合する", value=True)
                else:
                    force_str = st.checkbox("キーの型を強制的に「文字列」に統一して結合する", value=False)
                    
                if st.button(f"{merge_target_name} をマージ実行"):
                    try:
                        merge_df1 = df.copy()
                        merge_df2 = df_target.copy()

                        if force_str:
                            merge_df1[left_key] = merge_df1[left_key].astype(str)
                            merge_df2[right_key] = merge_df2[right_key].astype(str)

                        df_merged = pd.merge(merge_df1, merge_df2, left_on=left_key, right_on=right_key, how=how)
                        
                        base_name = st.session_state.output_name.replace('.csv', '')
                        add_name = merge_target_name.replace('.csv', '')
                        st.session_state.output_name = f"{base_name}_{add_name}.csv"
                        
                        st.session_state.df = df_merged
                        st.session_state.merge_success = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"結合エラーが発生しました。詳細: {e}")
                        
            if st.session_state.merge_success:
                st.success("✅ 結合が成功しました！下のボタンからすぐにダウンロードできます。")
                csv = st.session_state.df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇ 結合済みデータをダウンロード",
                    data=csv,
                    file_name=st.session_state.output_name,
                    mime='text/csv',
                )
        else:
            st.warning("結合機能を使うには、最初の画面でCSVを2つ以上アップロードしてください。")

    with tabs[10]:
        st.write("現在のデータフレームをダウンロード")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ CSV保存",
            data=csv,
            file_name=st.session_state.output_name,
            mime='text/csv',
        )
else:
    st.info("上にCSVファイルをアップロード（または選択）して開始してください。")
