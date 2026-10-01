import streamlit as st
import pandas as pd
import numpy as np
import gc
import warnings
import io

# 警告メッセージを画面に出さないようにする
warnings.filterwarnings('ignore')

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

# --- データ軽量化（ダウンキャスト）関数 ---
def reduce_mem_usage(df):
    """メモリを極限まで削減する処理"""
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)[:3] == 'int' or str(col_type)[:5] == 'float':
            c_min, c_max = df[col].min(), df[col].max()
            
            if pd.isna(c_min) or pd.isna(c_max): 
                continue 
                
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
    gc.collect() 
    return df

# --- CSV/Parquetの読み込みを自動判別する関数 ---
def load_dataframe(file, max_rows, sampling):
    # Parquetの場合
    if file.name.lower().endswith(('.parquet', '.pq')):
        df = pd.read_parquet(file)
        if sampling == "10%":
            df = df.sample(frac=0.1, random_state=42)
        elif sampling == "1%":
            df = df.sample(frac=0.01, random_state=42)
        return df.head(int(max_rows))
    # CSVの場合
    else:
        actual_nrows = int(max_rows)
        if sampling == "10%":
            actual_nrows = max(1, int(max_rows * 0.1))
        elif sampling == "1%":
            actual_nrows = max(1, int(max_rows * 0.01))
        return pd.read_csv(file, nrows=actual_nrows, low_memory=False)

# --- セッションステート初期化 ---
if "df" not in st.session_state:
    st.session_state.df = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "merge_success" not in st.session_state:
    st.session_state.merge_success = False
if "output_name" not in st.session_state:
    st.session_state.output_name = "data.csv"

st.title("Kaggle前処理捜査ファイル")
st.write("※メモリ不足エラーを防ぐため、100MB以上のファイルは「間引き: 10%以下」または「Parquet形式」を推奨します")

st.markdown("### ファイルを選択（CSV / Parquet対応）")
uploaded_raw_files = st.file_uploader("複数ファイル可・大容量はストリーミング読込", accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    max_rows = st.number_input("最大行数/ファイル", value=1000000, step=100000)
with col2:
    sampling = st.selectbox("間引き", ["全行", "10%", "1%"])

if uploaded_raw_files:
    uploaded_files = [f for f in uploaded_raw_files if f.name.lower().endswith(('.csv', '.parquet', '.pq', ''))]
    
    if not uploaded_files:
        st.warning("有効なCSVまたはParquetファイルを選択してください。")
    else:
        file_names = [f.name for f in uploaded_files]
        selected_file_name = st.radio("ベースとなるファイルを選択:", file_names, horizontal=True)
        
        if st.session_state.current_file != selected_file_name:
            selected_file = next(f for f in uploaded_files if f.name == selected_file_name)
            
            temp_df = load_dataframe(selected_file, max_rows, sampling)
            optimized_df = reduce_mem_usage(temp_df)
            
            st.session_state.df = optimized_df
            st.session_state.current_file = selected_file_name
            st.session_state.merge_success = False
            
            base_name = selected_file_name.rsplit('.', 1)[0]
            st.session_state.output_name = f"{base_name}.csv"
            
            gc.collect()
            st.rerun()

        df = st.session_state.df
        st.write(f"現在のデータ ({st.session_state.output_name}): {df.shape[0]:,}行 × {df.shape[1]}列")

        tabs = st.tabs(["先頭5行", "基本統計", "欠損", "ヒストグラム", "文字列カウント", "相関係数", "時系列", "クリーニング", "エンコード", "結合", "保存(形式選択)"])
        
        with tabs[0]:
            st.write(f"先頭5行 (全{df.shape[0]:,}行)")
            st.dataframe(df.head().astype(str), width='stretch')
            
        with tabs[1]:
            st.write("基本統計量")
            st.dataframe(df.describe(include='all').T.astype(str), width='stretch')
            
        with tabs[2]:
            st.write("列ごとの欠損値数")
            missing = df.isnull().sum().reset_index()
            missing.columns = ['列名', '欠損数']
            missing = missing[missing['欠損数'] > 0].sort_values('欠損数', ascending=False)
            if missing.empty:
                st.success("欠損値はありません。")
            else:
                st.dataframe(missing.astype(str), width='stretch')
                
        with tabs[3]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("列を選択 (ヒストグラム)", numeric_cols)
                st.bar_chart(df[selected_col].value_counts(bins=20, sort=False))
            else:
                st.warning("数値列がありません。")

        with tabs[4]:
            st.markdown("### 文字列（カテゴリ）のカウント")
            cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            if cat_cols:
                selected_cat_col = st.selectbox("集計する列を選択してください", cat_cols)
                count_df = df[selected_cat_col].value_counts().reset_index()
                count_df.columns = [selected_cat_col, '出現回数']
                st.dataframe(count_df.astype(str), width='stretch')
            else:
                st.info("文字列（カテゴリ）の列が見つかりません。")

        with tabs[5]:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.dataframe(numeric_df.corr().style.background_gradient(cmap='coolwarm'), width='stretch')
            else:
                st.warning("数値列がありません。")

        with tabs[6]:
            st.markdown("### 時間特徴量の自動抽出")
            date_cols = df.select_dtypes(include=['object', 'datetime', 'string']).columns.tolist()
            if date_cols:
                target_date_col = st.selectbox("日付が入力されている列を選択", date_cols)
                if st.button("時間特徴量を生成"):
                    try:
                        df[target_date_col] = pd.to_datetime(df[target_date_col])
                        df['year'] = df[target_date_col].dt.year
                        df['month'] = df[target_date_col].dt.month
                        df['day'] = df[target_date_col].dt.day
                        df['dayofweek'] = df[target_date_col].dt.dayofweek
                        st.session_state.df = df
                        st.success("時間特徴量を追加しました！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"エラー: {e}")

        with tabs[7]:
            st.markdown("### データのクリーニング")
            clean_option = st.radio("操作を選択:", ["列の削除", "欠損行の削除"], horizontal=True)

            if clean_option == "列の削除":
                cols_to_drop = st.multiselect("削除する不要な列を選択してください", df.columns.tolist())
                if st.button("列を削除"):
                    if cols_to_drop:
                        st.session_state.df = df.drop(columns=cols_to_drop)
                        st.rerun()
            
            elif clean_option == "欠損行の削除":
                missing_rows = df.isnull().any(axis=1).sum()
                st.write(f"欠損を含む行数: {missing_rows:,}行")
                if missing_rows > 0 and st.button("すべて削除"):
                    st.session_state.df = df.dropna()
                    st.rerun()

        with tabs[8]:
            st.markdown("### ワンホットエンコーディング")
            cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            if cat_cols:
                target_cols = st.multiselect("エンコードする列を選択してください", cat_cols)
                if st.button("実行"):
                    if target_cols:
                        st.session_state.df = pd.get_dummies(df, columns=target_cols, drop_first=False)
                        st.rerun()

        with tabs[9]:
            st.markdown("### データのマージ (キー結合)")
            available_files = [f for f in uploaded_files if f.name != selected_file_name]
            
            if available_files:
                merge_target_name = st.selectbox("結合する追加ファイルを選択", [f.name for f in available_files])
                
                if merge_target_name:
                    target_file = next(f for f in available_files if f.name == merge_target_name)
                    
                    df_target_raw = load_dataframe(target_file, max_rows, sampling="全行")
                    df_target = reduce_mem_usage(df_target_raw)
                    
                    col1_m, col2_m, col3_m = st.columns(3)
                    with col1_m:
                        left_key = st.selectbox("ベースデータのキー列", df.columns.tolist())
                    with col2_m:
                        right_key = st.selectbox(f"{merge_target_name}のキー列", df_target.columns.tolist())
                    with col3_m:
                        how = st.selectbox("結合方法", ["left", "inner", "outer", "right"])
                    
                    force_str = st.checkbox("キーの型を強制的に「文字列」に統一して結合する", value=True)
                        
                    if st.button(f"{merge_target_name} をマージ実行"):
                        try:
                            if force_str:
                                df[left_key] = df[left_key].astype(str)
                                df_target[right_key] = df_target[right_key].astype(str)

                            df_merged = pd.merge(df, df_target, left_on=left_key, right_on=right_key, how=how)
                            
                            base_name = st.session_state.output_name.replace('.csv', '')
                            add_name = merge_target_name.rsplit('.', 1)[0]
                            st.session_state.output_name = f"{base_name}_{add_name}.csv"
                            
                            st.session_state.df = df_merged
                            st.session_state.merge_success = True
                            
                            del df_target
                            gc.collect()
                            
                            st.rerun()
                        except Exception as e:
                            st.error(f"結合エラーが発生しました。詳細: {e}")
                            
                if st.session_state.merge_success:
                    st.success("✅ 結合成功！右の「保存」タブからダウンロードできます。")
            else:
                st.warning("結合機能を使うには、最初の画面でファイルを2つ以上アップロードしてください。")

        with tabs[10]:
            st.write("現在のデータをダウンロードします。お好きな形式を選んでください。")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇ CSVで保存 (汎用)",
                    data=csv_data,
                    file_name=st.session_state.output_name,
                    mime='text/csv',
                    use_container_width=True
                )
                
            with col_dl2:
                parquet_buffer = io.BytesIO()
                df.to_parquet(parquet_buffer, index=False)
                parquet_name = st.session_state.output_name.replace('.csv', '.parquet')
                st.download_button(
                    label="⬇ Parquetで保存 (軽量・高速)",
                    data=parquet_buffer.getvalue(),
                    file_name=parquet_name,
                    mime='application/octet-stream',
                    use_container_width=True
                )
else:
    st.info("上にCSVまたはParquetファイルをアップロードして開始してください。")
