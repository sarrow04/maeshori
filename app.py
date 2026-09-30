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

# --- セッションステートの初期化 ---
if "df" not in st.session_state:
    st.session_state.df = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

st.title("Kaggle前処理捜査ファイル")
st.write("CSVを読み込み → 基本統計・欠損・分布・相関を確認 (端末内で処理、送信なし)")

# --- ファイルアップロード領域 ---
st.markdown("### CSVを選択 / ここにドロップ")
uploaded_files = st.file_uploader("複数ファイル可・大容量はストリーミング読込", type=["csv"], accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    max_rows = st.number_input("最大行数/ファイル", value=1000000, step=100000)
with col2:
    sampling = st.selectbox("間引き", ["全行", "10%", "1%"])

if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    selected_file_name = st.radio("ファイルを選択:", file_names, horizontal=True)
    
    # 選択ファイルが切り替わった時のみデータを読み込み直し、状態をリセットする
    if st.session_state.current_file != selected_file_name:
        selected_file = next(f for f in uploaded_files if f.name == selected_file_name)
        
        if sampling == "10%":
            temp_df = pd.read_csv(selected_file, nrows=max_rows)
            st.session_state.df = temp_df.sample(frac=0.1, random_state=42)
        elif sampling == "1%":
            temp_df = pd.read_csv(selected_file, nrows=max_rows)
            st.session_state.df = temp_df.sample(frac=0.01, random_state=42)
        else:
            st.session_state.df = pd.read_csv(selected_file, nrows=max_rows)
            
        st.session_state.current_file = selected_file_name
        st.rerun()

    # 以降の処理はセッションステートに保存されたデータフレームを使用する
    df = st.session_state.df
    st.write(f"{selected_file_name}: {df.shape[0]:,}行 × {df.shape[1]}列")

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
        st.write("※ 未実装 (テキスト処理拡張用)")

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
        st.write("※ 未実装 (欠損値補完などの拡張用)")

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
        if len(uploaded_files) > 1:
            other_files = [f.name for f in uploaded_files if f.name != selected_file_name]
            merge_target_name = st.selectbox("結合するファイルを選択", other_files)
            
            if merge_target_name:
                target_file = next(f for f in uploaded_files if f.name == merge_target_name)
                df_target = pd.read_csv(target_file)
                
                col1_m, col2_m, col3_m = st.columns(3)
                with col1_m:
                    left_key = st.selectbox("現在のデータのキー列", df.columns.tolist())
                with col2_m:
                    right_key = st.selectbox("結合ファイルのキー列", df_target.columns.tolist())
                with col3_m:
                    how = st.selectbox("結合方法", ["left", "inner", "outer", "right"])
                    
                if st.button("マージを実行"):
                    try:
                        df_merged = pd.merge(df, df_target, left_on=left_key, right_on=right_key, how=how)
                        st.session_state.df = df_merged
                        st.success(f"（**{merge_target_name}**）を {how} 結合しました！ (現在 {df_merged.shape[0]:,}行 × {df_merged.shape[1]}列)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"結合エラーが発生しました。詳細: {e}")
        else:
            st.warning("結合機能を使うには、最初の画面でCSVを2つ以上ドロップしてください。")

    with tabs[10]:
        st.write("現在のデータフレームをダウンロード")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ CSV保存",
            data=csv,
            file_name=f"processed_{selected_file_name}",
            mime='text/csv',
        )
else:
    st.info("上にCSVファイルをドロップして開始してください。")
