# =====================================================================
# 拡張版 CSV前処理アプリ v5（Streamlit）
# ---------------------------------------------------------------------
# v5 の変更点（v4からの差分）
#  1. 日本語フォントの準備を強化。Streamlit Cloud（Python 3.12以降）では japanize_matplotlib が
#     動かず、グラフやA4サマリーPDFの日本語が「□」になるため、packages.txt で入れた Noto CJK を直接登録する
#  2. 「📄 A4サマリー」タブを追加：PDFを作らずに、④目的変数×時系列などの図をその場で画面に表示（スマホ向け）
# ---------------------------------------------------------------------
# v4 の変更点（v3からの差分）
#  1. 🤖機械学習タブの結果を「実験記録（1行のJSON）」で出力。CV・モデル・使ったカラム名・除外列・
#     重要度・設定が入り、実験捜査ファイル（記録ページ）に貼ると貯まる。v39ノートと同じ形式
#  2. 実験履歴をまとめて JSONL でダウンロード（記録ページの「ファイル」で一括登録できる）
#  3. 「データ名」を入力できるように（記録をデータごとに分ける。一度入れたらアプリを開いている間は覚えておく）
# ---------------------------------------------------------------------
# v3 の変更点（v2からの差分）
#  1. 「📄 A4サマリーPDF」を追加：データ基本情報サマリー(A4) v7 ノートと同じPDFを、アプリから直接作成・DL
#     （①基本情報表 ②グラフ一覧 ③相関 ④目的変数×時系列 ⑤目的変数との関係）。
#     目的変数・日付列は、左の「🎯目的変数」「📅日付列」をそのまま使う
# ---------------------------------------------------------------------
# v2 の変更点（v1 = 共有いただいた版からの差分）
#  【不具合の修正】
#   1. 行を消す処理（欠損行削除・IQR異常値・重複・数値/日付/カテゴリフィルタ）で
#      test の行まで消えていた → 既定で「test の行は消さない」。提出行数がずれなくなる
#   2. 「欠損行を削除」で目的変数の欠損行を消すと、test（目的変数が空）が全部消えていた
#   3. IQR異常値削除が 0/1 フラグ列や ID 列にもかかり、フラグ=1 の行が丸ごと消えていた
#      → 連続値の列だけ対象。四分位は train だけで計算し、列ごとに順番に削らず一括判定
#   4. 「文字列の不要な空白・改行を整理」がチェックしても実行されていなかった
#   5. 「すべて自動」のダミー化/ID化/category化で、日付の列まで変換・削除されていた
#      → 日付らしい列は自動対象から除外。ダミー化はユニーク数が多すぎる列も除外
#   6. 自動ダミー化の「出現率が低いフラグを削除」が、列名の部分一致で関係ない列まで消す恐れ
#      → 今回作ったダミー列だけを対象に変更
#   7. 「GZIP圧縮CSV」が実際には圧縮されていなかった／CSVにBOMが付かずExcelで文字化け
#   8. 目的変数の候補が見つからないとき、先頭列（日付など）が目的変数として守られていた → (なし) に
#   9. 文字数カウントで空欄が「nan」の3文字として数えられていた
#  10. 数値フィルタに数字以外を入れるとアプリが止まっていた
#  11. ログが記録されていなかった → 処理ごとの行数・列数を表示
#  12. カスタム境界値のラベルが小数を切り捨てて重複し、エラーになることがあった
#  13. ラグ・過去平均が「読み込み順」で計算されていた → 日付列を選ぶと日付順で計算
#  14. 型変換が STEP1 の後に動いていた → 最初に実行（型をそろえてからカテゴリ処理）
#  【機能の追加】
#  15. 提出見本（sample）の読み込み。ZIPに sample が入っていれば自動で認識
#  16. 🤖 機械学習タブ：v38ノートと同じ時系列CV・同じモデルで、その場でスコアを確認。
#      上位K平均・重要度・提出CSVダウンロード・実験履歴（1回1変更の記録）つき
#  17. ダウンロードZIPに sample を同梱し、v38ノートの設定セルに貼る設定文を表示
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import gzip
import zipfile
import itertools
import datetime as _dt
import json

from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# ---- 日本語フォント（グラフ・PDFの文字化け対策） ----
# packages.txt に fonts-noto-cjk を書くと Streamlit Cloud に Noto CJK が入るので、それを直接登録する。
# 見つからないときだけ japanize_matplotlib を試す（Python 3.12以降では動かないことがある）。
from matplotlib import font_manager as _fm
_jp_font = None
for _p in _fm.findSystemFonts():
    if "NotoSansCJK" in _p or "NotoSerifCJK" in _p or "ipaexg" in _p.lower():
        try: _fm.fontManager.addfont(_p)
        except Exception: pass
for _n in ["Noto Sans CJK JP", "IPAexGothic", "Noto Sans CJK SC"]:
    if any(f.name == _n for f in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _n; _jp_font = _n; break
if _jp_font is None:
    try:
        import japanize_matplotlib  # noqa: F401
        _jp_font = "IPAexGothic"
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# 設定（アプリ全体の既定値。画面で変えられないものだけここに置いています）
# =====================================================================

# 【設定1】何の設定か：目的変数（予測したい列）を自動で選ぶときの候補の列名。
# 入れ方：列名の文字列のリスト。前にあるものほど優先。 例 ["y", "賃料", "price"]
# 空([])のとき：自動選択せず、目的変数は「(なし)」から始まります（画面で選べばOK）。
TARGET_CANDIDATES = ["賃料", "目的変数", "target", "TARGET", "price", "SalePrice", "y", "sales", "kcal"]

# 【設定2】何の設定か：日付列（時系列の並び順に使う列）を自動で選ぶときの候補。
# 入れ方：列名のリスト。 例 ["datetime", "date", "日付"]
# 空([])のとき：日付列は「(なし)」から始まり、行の並び順を時間順とみなします。
DATE_CANDIDATES = ["datetime", "date", "Date", "日付", "年月日", "timestamp"]

# 【設定3】何の設定か：「全自動ダミー化」の対象にする列の、ユニーク数（種類数）の上限の既定値。
#   種類が多すぎる列（名前・住所など）をダミー化すると列が爆発するので除外します。
# 入れ方：整数。 例 50 → 51種類以上ある列はダミー化しない。画面でも変えられます。
# 空にはできません（0にすると全列が対象外）。
AUTO_DUMMY_MAX_UNIQUE = 50

# 【設定4】何の設定か：「日付らしい列」と判定する割合。文字列の列で、値の何割が
#   「2014-10-01」「2014/10/1」「2014年10月」のような形なら日付とみなすか。
#   日付らしい列は、全自動のダミー化・ID化・category化から外れます（日付が消えないように）。
# 入れ方：0〜1の小数。 例 0.8 → 8割以上が日付の形なら日付扱い。
# 空にはできません。1.0にすると、全部が日付の形のときだけ日付扱い。
DATE_LIKE_RATIO = 0.8

# 【設定5】何の設定か：機械学習タブで、学習に使う最大行数。Streamlit Community Cloud は
#   メモリが小さいので、大きいデータは末尾（新しい方）のこの行数だけで学習します。
# 入れ方：整数。 例 50000。
# 空(None)のとき：全行で学習します（重いデータだとアプリが落ちることがあります）。
ML_MAX_TRAIN_ROWS = 50000

# 【設定6】何の設定か：乱数の種。同じ値なら、同じデータ・同じ設定で毎回同じ結果になります。
# 入れ方：整数。 例 42（v38ノートと同じ）。
RANDOM_STATE = 42

st.set_page_config(page_title="拡張版 CSV前処理アプリ v5", layout="wide")

# =====================================================================
# 共通処理
# =====================================================================
DATE_RE = r"^\s*\d{4}\s*[-/年.]\s*\d{1,2}"


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
    elif fname_lower.endswith('.gz'):
        try: return pd.read_csv(file_bytes, compression='gzip', low_memory=False)
        except Exception as e: st.error(f"GZIP読込エラー: {e}"); st.stop()
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc-jp"]:
        try:
            if hasattr(file_bytes, 'seek'): file_bytes.seek(0)
            return pd.read_csv(file_bytes, encoding=enc, low_memory=False)
        except Exception: pass
    for enc in ["cp932", "utf-8"]:
        try:
            if hasattr(file_bytes, 'seek'): file_bytes.seek(0)
            return pd.read_csv(file_bytes, encoding=enc, encoding_errors='replace', low_memory=False)
        except Exception: pass
    try:
        if hasattr(file_bytes, 'seek'): file_bytes.seek(0)
        return pd.read_parquet(file_bytes)
    except Exception: pass
    st.error(f"ファイルを読み込めませんでした: {file_name}")
    st.stop()


def read_sample_bytes(raw, name, has_header):
    """提出見本を読む。お弁当の sample.csv はヘッダーなし（日付, 予測値）。"""
    if name.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, header=0 if has_header else None, low_memory=False)
        except Exception:
            pass
    return None


def clean_text_value(value):
    if pd.isna(value): return value
    s = str(value).replace("　", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()


def to_text(s):
    """欠損を空文字にして文字列化（category型でもエラーにならない）。"""
    return s.astype(object).where(s.notna(), "").astype(str)


def is_text_col(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def is_date_like(s):
    if pd.api.types.is_datetime64_any_dtype(s): return True
    if not is_text_col(s): return False
    v = s.dropna().astype(str).head(500)
    return len(v) > 0 and v.str.match(DATE_RE).mean() >= DATE_LIKE_RATIO


def get_numeric_columns(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != "_source_"]


def is_flag_series(s):
    vals = set(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
    return len(vals) > 0 and vals.issubset({0, 1})


def source_of(df):
    return df["_source_"] if "_source_" in df.columns else pd.Series("single", index=df.index)


def train_mask(df):
    return source_of(df) != "test"


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
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            row.update({"平均": s.mean(), "最小値": s.min(), "最大値": s.max()})
        else:
            row.update({"平均": np.nan, "最小値": np.nan, "最大値": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def to_parquet_bytes(df):
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, index=False)
    except Exception:
        # 1つの列に数値と文字が混ざっているとParquetにできないので、その列を文字列にする
        df2 = df.copy()
        for c in df2.columns:
            if pd.api.types.is_object_dtype(df2[c]):
                df2[c] = df2[c].astype(object).where(df2[c].notna(), None).map(lambda v: None if v is None else str(v))
        buf = io.BytesIO(); df2.to_parquet(buf, index=False)
    return buf.getvalue()


def convert_df(df, format_type, header=True):
    if format_type == "Parquet形式 (.parquet)":
        return to_parquet_bytes(df), "parquet", "application/octet-stream"
    csv_bytes = df.to_csv(index=False, header=header).encode("utf-8-sig")  # BOM付き → Excelで文字化けしない
    if format_type == "GZIP圧縮CSV (.csv.gz)":
        return gzip.compress(csv_bytes), "csv.gz", "application/gzip"
    return csv_bytes, "csv", "text/csv"


def add_log(log, df, step, detail=""):
    src = source_of(df)
    log.append({"処理": step, "内容": detail, "train行": int((src != "test").sum()),
                "test行": int((src == "test").sum()), "列数": int(df.shape[1] - (1 if "_source_" in df.columns else 0))})


def drop_rows(df, drop_mask, protect_test, log, step):
    """drop_mask=True の行を削除。protect_test=True なら test の行は残す。"""
    drop_mask = drop_mask.reindex(df.index).fillna(False).astype(bool)
    if protect_test:
        drop_mask &= train_mask(df)
    n = int(drop_mask.sum())
    df = df.loc[~drop_mask]
    add_log(log, df, step, f"{n}行削除" + ("（testは保護）" if protect_test else ""))
    return df


def time_sorted_index(df, date_col):
    """日付列があれば日付順（同じ日付は元の順）、なければ元の順の index を返す。"""
    if date_col and date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce")
        return d.sort_values(kind="stable", na_position="last").index
    return df.index


# =====================================================================
# 前処理の本体（UIから受け取った設定 P で実行）
# =====================================================================
def run_preprocess(df, P):
    df = df.copy()
    orig = df.copy()  # STEP3で、ダミー化等で消えた列も元の値から使えるように保持
    log = []
    target = P["target"]
    protect = [target, "_source_"]
    add_log(log, df, "開始")

    # ---- 0. 文字列の整理（v1では実行されていなかった） ----
    if P["clean_text"]:
        cols = [c for c in df.columns if c not in protect and is_text_col(df[c]) and not isinstance(df[c].dtype, pd.CategoricalDtype)]
        for c in cols: df[c] = df[c].map(clean_text_value)
        add_log(log, df, "文字列の整理", f"{len(cols)}列")

    # ---- 1. 型変換（最初に実行） ----
    for c in P["type_conv_cols"]:
        if c not in df.columns: continue
        t = P["type_conv_target"]
        if t == "文字列 (str)": df[c] = df[c].astype(object).where(df[c].notna(), np.nan).map(lambda v: v if pd.isna(v) else str(v))
        elif t == "整数 (int)": df[c] = pd.to_numeric(df[c], errors='coerce').round().astype('Int64')
        elif t == "小数 (float)": df[c] = pd.to_numeric(df[c], errors='coerce')
        elif t == "日付 (datetime)": df[c] = pd.to_datetime(df[c], errors='coerce')
    if P["type_conv_cols"]: add_log(log, df, "型変換", f"{P['type_conv_cols']} → {P['type_conv_target']}")

    # ---- 2. 複数列に分割 ----
    sc = P["split_col"]
    if sc != "(なし)" and sc in df.columns:
        prefix = f"{sc}_分割"
        temp_s = to_text(df[sc])
        delim = P["split_delim"]
        auto = delim.startswith("自動認識")
        actual = '|||' if auto or delim == "空白" else delim
        if auto: temp_s = temp_s.str.replace(r'[／/｜|\[\]［］\s]+', '|||', regex=True).str.strip('|')
        elif delim == "空白": temp_s = temp_s.str.replace(r'\s+', '|||', regex=True).str.strip('|')
        if P["split_strategy"].startswith("単語ごとに"):
            dum = temp_s.str.get_dummies(sep=actual).drop(columns=["", "nan"], errors="ignore")
            for c in dum.columns: df[f"{prefix}_{c}"] = dum[c].map({1: c, 0: np.nan})
        else:
            lst = temp_s.apply(lambda x: [p.strip() for p in str(x).split(actual) if p.strip()])
            for i in range(int(P["split_max"])): df[f"{prefix}_{i+1}"] = lst.apply(lambda x: x[i] if i < len(x) else np.nan)
        if P["split_drop"]: df = df.drop(columns=[sc])
        add_log(log, df, "複数列に分割", sc)

    # ---- 3. 複数キーワードの一括フラグ化 ----
    mc = P["ml_column"]
    if mc != "(なし)" and mc in df.columns:
        temp_s = to_text(df[mc])
        delim = '|' if P["ml_delim"] == "空白" else (P["ml_delim"] or "/")
        if delim == '|': temp_s = temp_s.str.strip().str.replace(r'\s+', '|', regex=True)
        dum = temp_s.str.get_dummies(sep=delim).astype(int).drop(columns=["", "nan"], errors="ignore")
        dum.columns = [f"{mc}_{c.strip()}" for c in dum.columns]
        dum = dum.T.groupby(level=0).max().T  # 前後の空白違いで同名になった列をまとめる
        df = pd.concat([df, dum], axis=1)
        if P["ml_drop"]: df = df.drop(columns=[mc])
        add_log(log, df, "キーワード一括フラグ", f"{mc}: {dum.shape[1]}列追加")

    # ---- 4. ダミー化 ----
    if P["dummy_auto"]:
        cands = [c for c in df.columns if c not in protect and is_text_col(df[c]) and not is_date_like(df[c])]
        valid = [c for c in cands if df[c].isna().mean() * 100 <= P["dummy_max_missing"]
                 and df[c].nunique(dropna=True) <= P["dummy_max_unique"]]
        skipped = [c for c in cands if c not in valid]
        if valid:
            before = set(df.columns)
            df = pd.get_dummies(df, columns=valid, dummy_na=False, dtype=int)
            new_cols = [c for c in df.columns if c not in before]
            drop_dums = [c for c in new_cols if df[c].mean() * 100 < P["dummy_min_freq"]]
            if drop_dums: df = df.drop(columns=drop_dums)
            add_log(log, df, "自動ダミー化", f"対象{valid} / 作成{len(new_cols)}列・低頻度で{len(drop_dums)}列削除 / 対象外{skipped}")
    elif P["onehot_columns"]:
        cols = [c for c in P["onehot_columns"] if c in df.columns]
        df = pd.get_dummies(df, columns=cols, dummy_na=False, dtype=int)
        add_log(log, df, "ダミー化", str(cols))

    # ---- 5. category 型 ----
    if P["cat_auto"]:
        cat_cols = [c for c in df.columns if c not in protect and is_text_col(df[c])
                    and not isinstance(df[c].dtype, pd.CategoricalDtype) and not is_date_like(df[c])]
    else:
        cat_cols = [c for c in P["cat_columns"] if c in df.columns]
    for c in cat_cols: df[c] = df[c].astype("category")
    if cat_cols: add_log(log, df, "category型に変換", str(cat_cols))

    # ---- 6. Label Encoding ----
    if P["le_auto"]:
        le_cols = [c for c in df.columns if c not in protect and is_text_col(df[c]) and not is_date_like(df[c])]
    else:
        le_cols = [c for c in P["le_columns"] if c in df.columns]
    for c in le_cols:
        codes = pd.factorize(df[c])[0].astype(float)
        codes[codes < 0] = np.nan
        df[f"{c}_ID"] = codes
        if P["le_drop"]: df = df.drop(columns=[c])
    if le_cols: add_log(log, df, "数値ID化", str(le_cols))

    # ---- 7. 欠損値 ----
    ma = P["missing_action"]
    is_tr = train_mask(df)
    if ma == "数値列を0で補完":
        cols = [c for c in get_numeric_columns(df) if c != target and df[c].isna().any()]
        for c in cols: df[c] = df[c].fillna(0)
        add_log(log, df, "欠損を0で補完", f"{len(cols)}列")
    elif ma == "数値列を中央値で補完(trainの中央値)":
        cols = [c for c in get_numeric_columns(df) if c != target and df[c].isna().any()]
        for c in cols: df[c] = df[c].fillna(df.loc[is_tr, c].median())
        add_log(log, df, "欠損を中央値で補完", f"{len(cols)}列")
    elif ma == "目的変数が欠損の行を削除(trainのみ)":
        if target: df = drop_rows(df, df[target].isna(), True, log, "目的変数の欠損行を削除")
    elif ma == "どこかに欠損がある行を削除(trainのみ)":
        cols = [c for c in df.columns if c not in protect]
        df = drop_rows(df, df[cols].isna().any(axis=1), True, log, "欠損がある行を削除")
    if P["fill_text_missing"]:
        cols = [c for c in df.columns if c not in protect and is_text_col(df[c]) and df[c].isna().any()]
        for c in cols:
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                if "不明" not in df[c].cat.categories: df[c] = df[c].cat.add_categories(["不明"])
            df[c] = df[c].fillna("不明")
        add_log(log, df, "文字列の欠損を「不明」で補完", f"{len(cols)}列")

    # ---- 8. 重複行 ----
    if P["duplicate"]:
        df = drop_rows(df, df.duplicated(), P["protect_test"], log, "重複行を削除")

    # ---- 9. IQR 異常値（連続値の列だけ・四分位はtrainで計算・一括判定） ----
    if P["outlier_action"] == "IQRで異常値行を削除":
        if P["iqr_cols"]:
            cols = [c for c in P["iqr_cols"] if c in df.columns]
        else:
            cols = [c for c in get_numeric_columns(df) if c != target and not str(c).endswith("_ID")
                    and not is_flag_series(df[c]) and df[c].nunique(dropna=True) > 10]
        is_tr = train_mask(df)
        out = pd.Series(False, index=df.index)
        for c in cols:
            v = pd.to_numeric(df[c], errors="coerce")
            s = v[is_tr].dropna()
            if len(s) < 4: continue
            q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
            if iqr == 0: continue
            out |= (v < q1 - P["iqr_k"] * iqr) | (v > q3 + P["iqr_k"] * iqr)
        df = drop_rows(df, out, True, log, f"IQR異常値を削除（{len(cols)}列を判定）")

    # ---- 10. STEP3 特徴量 ----
    df = run_feature(df, P, log, orig)

    # ---- 11. 列の削除 ----
    if P["drop_by_name"].strip():
        words = [w.strip() for w in P["drop_by_name"].split(",") if w.strip()]
        to_drop = [c for c in df.columns if c not in protect and any(w in str(c) for w in words)]
        df = df.drop(columns=to_drop); add_log(log, df, "列名で削除", str(to_drop))
    if P["min_count"] > 0:
        is_tr = train_mask(df)
        to_drop = [c for c in df.columns if c not in protect and pd.api.types.is_numeric_dtype(df[c])
                   and is_flag_series(df[c]) and df.loc[is_tr, c].sum() <= P["min_count"]]
        df = df.drop(columns=to_drop); add_log(log, df, f"1が{P['min_count']}件以下のフラグ列を削除", str(to_drop))
    if P["drop_columns"]:
        to_drop = [c for c in P["drop_columns"] if c in df.columns and c not in protect]
        df = df.drop(columns=to_drop); add_log(log, df, "手動で列削除", str(to_drop))

    # ---- 12. 行のフィルタ（列がダミー化等で消えていても、元の値で判定する） ----
    def colvals(c):
        return df[c] if c in df.columns else (orig[c].reindex(df.index) if c in orig.columns else None)

    if P["date_filter_col"] != "(なし)" and colvals(P["date_filter_col"]) is not None:
        d = pd.to_datetime(colvals(P["date_filter_col"]), errors="coerce")
        keep = pd.Series(True, index=df.index)
        dmin = pd.to_datetime(P["date_min"].strip(), errors="coerce") if P["date_min"].strip() else pd.NaT
        dmax = pd.to_datetime(P["date_max"].strip(), errors="coerce") if P["date_max"].strip() else pd.NaT
        if pd.notna(dmin): keep &= d >= dmin
        if pd.notna(dmax): keep &= d <= dmax
        df = drop_rows(df, ~keep, P["protect_test"], log, "日付で絞り込み")
    if P["range_column"] != "(なし)" and colvals(P["range_column"]) is not None:
        v = pd.to_numeric(colvals(P["range_column"]), errors="coerce")
        keep = pd.Series(True, index=df.index)
        try:
            if P["min_value"].strip(): keep &= v >= float(P["min_value"])
            if P["max_value"].strip(): keep &= v <= float(P["max_value"])
            df = drop_rows(df, ~keep, P["protect_test"], log, "数値で絞り込み")
        except ValueError:
            add_log(log, df, "数値で絞り込み", "⚠ 最小値/最大値が数字ではないためスキップ")
    if P["cat_filter_col"] != "(なし)" and P["cat_filter_vals"] and colvals(P["cat_filter_col"]) is not None:
        cv = colvals(P["cat_filter_col"]).astype(object)
        m = pd.Series(False, index=df.index)
        for val in P["cat_filter_vals"]:
            m |= cv.isna() if pd.isna(val) else (cv == val)
        df = drop_rows(df, m, P["protect_test"], log, "カテゴリ値で行削除")

    df = fix_dataframe_for_streamlit(df)
    add_log(log, df, "完了")
    return df, log


def run_feature(df, P, log, orig=None):
    m = P["fe_method"]; col = P["fe_column"]; target = P["target"]
    if m == "何もしない": return df
    before_cols = set(df.columns)
    # 基準の列がSTEP1（ダミー化・ID化など）で消えていたら、元データの値を一時的に借りる（行は index で対応）
    borrowed = []
    if orig is not None:
        for c in [col, P.get("fe_num_col"), P.get("fe_col2"), P.get("fe_lag_group")]:
            if c and c != "(なし)" and c not in df.columns and c in orig.columns:
                df[c] = orig[c].reindex(df.index); borrowed.append(c)

    if m == "【自動生成】数値カラム同士の総当たり割り算 (A ÷ B)":
        pfx = f"{P['fe_prefix'].strip()}_" if P["fe_prefix"].strip() else ""
        for c1, c2 in itertools.permutations([c for c in P["fe_auto_cols"] if c in df.columns], 2):
            s1 = pd.to_numeric(df[c1], errors="coerce"); s2 = pd.to_numeric(df[c2], errors="coerce")
            df[f"{pfx}{c1}÷{c2}"] = np.where(s2 == 0, np.nan, s1 / s2.replace(0, np.nan))

    elif m == "【時系列】カテゴリ別の過去実績(y)の平均 (ターゲットエンコーディング)":
        if target and col != "(なし)" and col in df.columns:
            pfx = P["fe_prefix"].strip() or f"{col}"
            n = int(P["fe_lag_n"])
            order = time_sorted_index(df, P["date_col"])
            tmp = df.loc[order]
            y = pd.to_numeric(tmp[target], errors="coerce")
            # 1つ前までの実績の平均（自分の行の答えは使わない）。testの行には直前の値を引き継ぐ
            res = y.groupby(tmp[col], observed=True).transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).mean().ffill())
            df[f"{pfx}_{target}_{n}回過去平均"] = res  # index で元の順に戻る

    elif col != "(なし)" and col in df.columns:
        source = df[col]
        prefix = P["fe_prefix"].strip() or f"{col}"

        if m == "【変換】カテゴリを数値IDに変換 (Label Encoding - 決定木向け)":
            codes = pd.factorize(source)[0].astype(float); codes[codes < 0] = np.nan
            df[f"{prefix}_ID"] = codes
        elif m == "【変換】カテゴリを頻度（出現回数）に変換 (Count Encoding)":
            df[f"{prefix}_頻度"] = source.astype(object).map(source.astype(object).value_counts(dropna=False)).astype(float)
        elif m == "【型変換】カテゴリ型 (category) に変換 (LightGBMネイティブ用)":
            df[f"{prefix}_cat"] = source.astype("category")
        elif m == "【変換】数値を任意の幅でグループ化 (例: 10刻みで「〇〇代」)":
            num_s = pd.to_numeric(source, errors="coerce"); w = P["fe_group_width"]
            df[f"{prefix}_{w}区切り"] = num_s.apply(lambda v: f"{int(np.floor(v / w) * w)}{P['fe_group_suffix']}" if pd.notna(v) else np.nan)
        elif m == "【変換】カスタム境界値で数値をカテゴリ化":
            try:
                bins = sorted(set(float(x.strip()) for x in P["fe_custom_bins"].split(",") if x.strip()))
                if bins:
                    edges = [-np.inf] + bins + [np.inf]
                    f = lambda v: f"{v:g}"
                    labels = [f"{f(bins[0])}未満"] + [f"{f(bins[i])}〜{f(bins[i+1])}" for i in range(len(bins) - 1)] + [f"{f(bins[-1])}以上"]
                    df[f"{prefix}_カテゴリ"] = pd.cut(pd.to_numeric(source, errors="coerce"), bins=edges, labels=labels, right=False)
            except Exception as e:
                add_log(log, df, "カスタム境界値", f"⚠ 失敗: {e}")
        elif m == "【変換】数値のビン分割 (10分割)":
            df[f"{prefix}_ビン"] = pd.qcut(pd.to_numeric(source, errors="coerce"), q=10, labels=False, duplicates='drop')
        elif m == "【時系列】日付列から年・月・日・曜日を抽出":
            d = pd.to_datetime(source, errors="coerce")
            df[f"{prefix}_年"] = d.dt.year; df[f"{prefix}_月"] = d.dt.month
            df[f"{prefix}_日"] = d.dt.day; df[f"{prefix}_曜日"] = d.dt.dayofweek
        elif m == "【時系列】過去の値を取得 (ラグ特徴量)":
            n = int(P["fe_lag_n"]); order = time_sorted_index(df, P["date_col"]); tmp = df.loc[order]
            g = P["fe_lag_group"]
            if g != "(なし)" and g in df.columns:
                df[f"{prefix}_{g}別_{n}ラグ"] = tmp.groupby(g, observed=True)[col].shift(n)
            else:
                df[f"{prefix}_{n}ラグ"] = tmp[col].shift(n)
        elif m == "【時系列】日付のフォーマット変換":
            fmt = {"yyyy-m-d": "%Y-%-m-%-d", "yyyy-MM-dd": "%Y-%m-%d", "yyyy/m/d": "%Y/%-m/%-d",
                   "yyyy/MM/dd": "%Y/%m/%d", "yyyyMMdd": "%Y%m%d"}[P["date_format"]]
            d = pd.to_datetime(source, errors="coerce")
            df[f"{prefix}_フォーマット済"] = d.map(lambda x: np.nan if pd.isna(x) else
                                              x.strftime(fmt.replace("%-m", str(x.month)).replace("%-d", str(x.day))))
        elif m == "【数値計算】2つの列で四則演算 (+, -, *, /)":
            c2 = P["fe_col2"]
            if c2 and c2 in df.columns:
                s1 = pd.to_numeric(df[col], errors="coerce"); s2 = pd.to_numeric(df[c2], errors="coerce")
                op = P["fe_op"]
                if op == "+": df[f"{prefix}_和_{c2}"] = s1 + s2
                elif op == "-": df[f"{prefix}_差_{c2}"] = s1 - s2
                elif op == "×": df[f"{prefix}_積_{c2}"] = s1 * s2
                elif op == "÷": df[f"{prefix}_商_{c2}"] = s1 / s2.replace(0, np.nan)
        elif m == "【集約】カテゴリごとの平均値などを算出 (複数選択可)":
            nc = P["fe_num_col"]
            if nc and nc != "(なし)" and nc in df.columns:
                is_tr = train_mask(df)
                num = pd.to_numeric(df[nc], errors="coerce")
                for agg in P["fe_aggs"]:
                    stat = num[is_tr].groupby(df.loc[is_tr, col].astype(object)).agg(agg)  # 集計は train だけで
                    df[f"{prefix}_{nc}_{agg}"] = df[col].astype(object).map(stat).astype(float)
        elif m == "【数値計算】対数変換 (Log1p - 歪み補正)":
            df[f"{prefix}_log1p"] = np.log1p(np.clip(pd.to_numeric(source, errors="coerce"), 0, None))
        elif m == "【フラグ】文字が入力されていれば1、空欄なら0 (有無フラグ)":
            t = to_text(source).str.strip()
            df[f"{prefix}_あり"] = ((t != "") & (t.str.lower() != "nan")).astype(int)
        elif m == "【フラグ】キーワードが含まれていれば1、なければ0":
            kws = [k.strip() for k in P["fe_keyword"].split(",") if k.strip()]
            if kws:
                suffix = P["fe_flag_suffix"].strip() or "フラグ"
                df[f"{prefix}_{suffix}"] = to_text(source).str.contains("|".join(map(re.escape, kws)), na=False).astype(int)
        elif m == "【フラグ】複数ルールの専用フラグを一括生成 (テキスト入力)":
            s_str = to_text(source)
            for line in P["fe_dict_text"].split("\n"):
                if ":" in line or "：" in line:
                    d = ":" if ":" in line else "："
                    name, kws = line.split(d, 1)
                    kw_list = [k.strip() for k in re.split(r'[,、|，]', kws) if k.strip()]
                    if name.strip() and kw_list:
                        df[f"{prefix}_{name.strip()}"] = s_str.str.contains("|".join(map(re.escape, kw_list)), na=False).astype(int)
        elif m == "【抽出】「徒歩〇分」から最短分数を抽出":
            def walk(x):
                if pd.isna(x): return np.nan
                f = re.findall(r"徒歩\s*(\d+)\s*分?", str(x))
                return float(min(map(int, f))) if f else np.nan
            df[f"{prefix}_最短徒歩分"] = source.map(walk)
        elif m == "【抽出】「築〇年」から年数を抽出 (単一の数値に変換)":
            def get_age(s):
                if pd.isna(s): return np.nan
                s = str(s)
                if "新築" in s: return 0.0
                y_m = re.search(r"(\d+)\s*年", s); m_m = re.search(r"(\d+)\s*[ヶ箇カかケヵ]?月", s)
                if not y_m and not m_m: return np.nan
                return (int(y_m.group(1)) if y_m else 0) + ((int(m_m.group(1)) if m_m else 0) / 12.0)
            df[f"{prefix}_築年数"] = source.map(get_age)
        elif m == "【抽出】「〇年〇ヶ月」を年と月に別々の列に分割":
            def ym(x):
                if pd.isna(x): return (np.nan, np.nan)
                s = str(x)
                if "新築" in s: return (0.0, 0.0)
                y_m = re.search(r"(\d+)\s*年", s); m_m = re.search(r"(\d+)\s*[ヶ箇カかケヵ]?月", s)
                if not y_m and not m_m: return (np.nan, np.nan)
                return (float(y_m.group(1)) if y_m else 0.0, float(m_m.group(1)) if m_m else 0.0)
            pairs = source.map(ym)
            df[f"{prefix}_年"] = pairs.map(lambda p: p[0]); df[f"{prefix}_ヶ月"] = pairs.map(lambda p: p[1])
        elif m == "【抽出】最初の数値を抽出":
            df[f"{prefix}_数値"] = pd.to_numeric(to_text(source).str.replace(",", "", regex=False).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
        elif m == "【テキスト】文字数カウント":
            df[f"{prefix}_文字数"] = source.map(lambda x: np.nan if pd.isna(x) else float(len(str(x))))

    if borrowed: df = df.drop(columns=borrowed)
    new_cols = [c for c in df.columns if c not in before_cols]
    add_log(log, df, "特徴量作成", f"{m.split('】')[0]}】 → {new_cols}" if new_cols else f"{m}（追加なし：列の選択を確認）")
    return df


# =====================================================================
# 機械学習（v38ノートと同じ考え方：時系列CV → 上位K平均 → 末尾検証の重要度 → 全データで再学習）
# =====================================================================
SCALED_MODELS = {'Ridge', 'BayesianRidge', 'ElasticNet', 'KNN', 'SVR'}
NAN_OK = ('LightGBM', 'HistGradientBoosting')


class LevelAdjustedRegressor(RegressorMixin, BaseEstimator):
    """目的変数を「その頃の水準」で割る/引いて学習し、予測時に直近の水準で戻す（v38と同じ）。"""
    def __init__(self, estimator=None, mode='ratio', window=21, last_n=20):
        self.estimator = estimator; self.mode = mode; self.window = window; self.last_n = last_n

    def fit(self, X, y, **fit_params):
        yv = np.asarray(y, dtype=float)
        lv = (pd.Series(yv).rolling(int(self.window), center=True, min_periods=max(3, int(self.window) // 3))
              .mean().bfill().ffill().to_numpy())
        if self.mode == 'ratio':
            lv = np.where(np.abs(lv) < 1e-9, 1e-9, lv); t = yv / lv
        else:
            t = yv - lv
        self.end_level_ = float(yv[-int(self.last_n):].mean())
        self.est_ = clone(self.estimator).fit(X, t, **fit_params)
        return self

    def predict(self, X):
        p = np.asarray(self.est_.predict(X), dtype=float)
        return p * self.end_level_ if self.mode == 'ratio' else p + self.end_level_


class AverageEnsemble(RegressorMixin, BaseEstimator):
    def __init__(self, members=None):
        self.members = members

    def fit(self, X, y, **fit_params):
        for m_ in self.members: m_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return np.mean([m_.predict(X) for m_ in self.members], axis=0)


def model_candidates(metric):
    lgb_obj = 'regression' if metric == 'rmse' else 'regression_l1'
    gbm_loss = 'squared_error' if metric == 'rmse' else 'absolute_error'
    rs = RANDOM_STATE
    m = {}
    if HAS_LGBM:
        m['LightGBM | 主力'] = LGBMRegressor(objective=lgb_obj, n_estimators=150, learning_rate=0.03, num_leaves=10, max_depth=4,
                                           min_child_samples=15, random_state=rs, verbosity=-1, n_jobs=1)
        m['LightGBM | 浅い(深さ2・葉4)'] = LGBMRegressor(objective=lgb_obj, n_estimators=200, learning_rate=0.03, num_leaves=4, max_depth=2,
                                                   min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                   reg_lambda=5.0, random_state=rs, verbosity=-1, n_jobs=1)
        m['LightGBM | 葉15・深さ5'] = LGBMRegressor(objective=lgb_obj, n_estimators=200, learning_rate=0.03, num_leaves=15, max_depth=5,
                                                min_child_samples=10, random_state=rs, verbosity=-1, n_jobs=1)
    m['GBM | 浅い木(深さ2)'] = GradientBoostingRegressor(loss=gbm_loss, n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=rs)
    m['RandomForest'] = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=rs, n_jobs=1)
    m['ExtraTrees'] = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=rs, n_jobs=1)
    m['HistGradientBoosting'] = HistGradientBoostingRegressor(loss=gbm_loss, max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=10, random_state=rs)
    m['Ridge'] = Ridge(alpha=10.0)
    m['BayesianRidge'] = BayesianRidge()
    m['ElasticNet'] = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=3, max_iter=20000, random_state=rs)
    m['KNN'] = KNeighborsRegressor(n_neighbors=7, weights='distance')
    m['SVR'] = TransformedTargetRegressor(regressor=SVR(C=3.0, epsilon=0.1), transformer=StandardScaler())
    return m


def prepare_model_input(df_part, feat_cols):
    x = df_part[feat_cols].copy()
    for c in feat_cols:
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = ((x[c] - pd.Timestamp('1970-01-01')) / pd.Timedelta(days=1)).astype(float)
        elif pd.api.types.is_numeric_dtype(x[c]) or pd.api.types.is_bool_dtype(x[c]):
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype(float)
        else:
            x[c] = x[c].astype(object).where(x[c].notna(), '__MISSING__').astype(str)
    return x


def build_pipeline(name, model, X, feat_cols, level):
    cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    tfs = []
    if num_cols:
        if name in SCALED_MODELS:
            tf = Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
        else:
            tf = 'passthrough' if name.startswith(NAN_OK) else SimpleImputer(strategy='median')
        tfs.append(('num', tf, num_cols))
    if cat_cols:
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', max_categories=50, min_frequency=5, sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        tfs.append(('cat', ohe, cat_cols))
    pipe = Pipeline([('prep', ColumnTransformer(tfs, sparse_threshold=0.0)), ('model', clone(model))])
    if level["mode"]:
        return LevelAdjustedRegressor(pipe, mode=level["mode"], window=level["window"], last_n=level["last_n"])
    return pipe


def metric_values(y, p):
    return {'RMSE': float(np.sqrt(mean_squared_error(y, p))), 'MAE': float(mean_absolute_error(y, p)),
            'R2': float(r2_score(y, p)) if len(y) > 1 else np.nan}


def run_ml(df, C):
    """C: 機械学習タブの設定。結果を dict で返す（画面表示とダウンロードに使う）。"""
    target, date_col = C["target"], C["date_col"]
    src = source_of(df)
    train = df[src != "test"].copy()
    test = df[src == "test"].copy() if (src == "test").any() else None
    train[target] = pd.to_numeric(train[target], errors="coerce")
    n0 = len(train); train = train.dropna(subset=[target])
    msgs = []
    if len(train) < n0: msgs.append(f"目的変数が空/数値でない {n0 - len(train)} 行を学習から除外")
    if date_col:
        train[date_col] = pd.to_datetime(train[date_col], errors="coerce")
        train = train.sort_values(date_col, kind="stable")
        if test is not None: test[date_col] = pd.to_datetime(test[date_col], errors="coerce")  # testは元の順のまま（見本と行をそろえる）
    train = train.reset_index(drop=True)
    if ML_MAX_TRAIN_ROWS and len(train) > ML_MAX_TRAIN_ROWS:
        train = train.iloc[-ML_MAX_TRAIN_ROWS:].reset_index(drop=True)
        msgs.append(f"重いので末尾 {ML_MAX_TRAIN_ROWS} 行だけで学習")
    if test is not None and target in test.columns: test = test.drop(columns=[target])

    excluded = set(C["exclude"]) | {target, "_source_"}
    if date_col and not C["include_date"]: excluded.add(date_col)
    feat = [c for c in train.columns if c not in excluded and (test is None or c in test.columns)]
    if C["min_flag"]:
        rare = [c for c in feat if is_flag_series(train[c]) and int((pd.to_numeric(train[c], errors="coerce") == 1).sum()) < C["min_flag"]]
        feat = [c for c in feat if c not in rare]
        if rare: msgs.append(f"1が{C['min_flag']}行未満の0/1列を除外: {rare}")
    feat = [c for c in feat if train[c].notna().any()]
    if not feat: raise ValueError("学習に使える特徴量がありません。")

    X = prepare_model_input(train, feat); y = train[target].astype(float)
    n = len(train)
    horizon_req = C["horizon"] or (len(test) if test is not None else 20)
    horizon = min(int(horizon_req), max(1, n // (C["folds"] + 2)))
    starts = [n - horizon * (C["folds"] - i) for i in range(C["folds"])]
    starts = [s for s in starts if s >= max(10, horizon) and s + horizon <= n]
    splits = [(np.arange(0, s), np.arange(s, s + horizon)) for s in starts]
    if not splits: raise ValueError("時系列CVの分割が作れません（行が少なすぎます）。")

    cands = model_candidates(C["metric"])
    names = [m for m in C["models"] if m in cands] or list(cands)
    clip = (lambda p: np.maximum(0, p)) if C["clip0"] else (lambda p: p)
    rows, fold_pred, fold_truth = [], {}, {}
    for nm in names:
        for k, (ti, vi) in enumerate(splits, 1):
            pipe = build_pipeline(nm, cands[nm], X.iloc[ti], feat, C["level"])
            pipe.fit(X.iloc[ti], y.iloc[ti])
            p = clip(pipe.predict(X.iloc[vi]))
            fold_pred[(nm, k)] = p; fold_truth[k] = y.iloc[vi].to_numpy()
            rows.append({'モデル': nm, 'Fold': k, **metric_values(y.iloc[vi], p)})
    detail = pd.DataFrame(rows)
    key = 'RMSE' if C["metric"] == 'rmse' else 'MAE'
    summary = (detail.groupby('モデル', as_index=False)
               .agg(CV_RMSE平均=('RMSE', 'mean'), CV_MAE平均=('MAE', 'mean'), CV_ばらつき=(key, 'std'), CV_R2平均=('R2', 'mean'))
               .sort_values(f'CV_{key}平均').reset_index(drop=True))
    summary.insert(0, '順位', np.arange(1, len(summary) + 1))
    ranked = summary['モデル'].tolist()

    ens = []
    for k in range(1, len(ranked) + 1):
        vals = [metric_values(fold_truth[f], np.mean([fold_pred[(m, f)] for m in ranked[:k]], axis=0)) for f in fold_truth]
        ens.append({'K': k, '平均するモデル': ' + '.join(ranked[:k]),
                    'CV_RMSE平均': np.mean([v['RMSE'] for v in vals]), 'CV_MAE平均': np.mean([v['MAE'] for v in vals])})
    ens = pd.DataFrame(ens)
    k_use = int(min(C["top_k"], len(ranked))) if C["top_k"] else int(ens.sort_values(f'CV_{key}平均')['K'].iloc[0])
    final_names = ranked[:k_use]
    cv_score = float(ens.loc[ens['K'] == k_use, f'CV_{key}平均'].iloc[0])

    def make_ens(Xs):
        return AverageEnsemble([build_pipeline(m, cands[m], Xs, feat, C["level"]) for m in final_names])

    # 末尾検証（最後のhorizon行）：予測の形と重要度を見る
    ti, vi = splits[-1]
    hold = make_ens(X.iloc[ti]).fit(X.iloc[ti], y.iloc[ti])
    hold_pred = clip(hold.predict(X.iloc[vi]))
    hold_df = pd.DataFrame({'日付': train[date_col].iloc[vi].values if date_col else vi,
                            '実績': y.iloc[vi].values, '予測': hold_pred})
    imp = None
    if C["perm"]:
        pi = permutation_importance(hold, X.iloc[vi], y.iloc[vi], n_repeats=5, random_state=RANDOM_STATE,
                                    scoring='neg_root_mean_squared_error' if C["metric"] == 'rmse' else 'neg_mean_absolute_error')
        imp = (pd.DataFrame({'特徴量': feat, '重要度平均': pi.importances_mean, '重要度標準偏差': pi.importances_std})
               .sort_values('重要度平均', ascending=False).reset_index(drop=True))

    test_pred = None
    if test is not None and len(test):
        final = make_ens(X).fit(X, y)
        test_pred = clip(final.predict(prepare_model_input(test, feat)))

    return {"summary": summary, "ens": ens, "final_names": final_names, "k": k_use, "cv_score": cv_score, "key": key,
            "hold_df": hold_df, "imp": imp, "test_pred": test_pred, "test": test, "feat": feat, "msgs": msgs,
            "horizon": horizon, "n_folds": len(splits), "train_rows": n}


def build_submission(res, sample_df, target, date_col):
    pred = np.asarray(res["test_pred"], dtype=float)
    test = res["test"]
    if sample_df is not None:
        if len(sample_df) != len(pred):
            raise ValueError(f"提出見本は{len(sample_df)}行、testの予測は{len(pred)}行で一致しません。前処理でtestの行が減っていないか確認してください。")
        sub = sample_df.copy()
        if sub.shape[1] == 2 and all(isinstance(c, (int, np.integer)) for c in sub.columns):
            sub[sub.columns[1]] = pred; header = False  # ヘッダーなし2列（お弁当形式）
        else:
            col = next((c for c in [target, 'y', 'target', 'sales', 'prediction', 'pred'] if c in sub.columns), sub.columns[-1])
            sub[col] = pred; header = True
    else:
        sub = pd.DataFrame({date_col: test[date_col].dt.strftime("%Y-%m-%d").values, target: pred}) if date_col and date_col in test.columns \
            else pd.DataFrame({target: pred})
        header = sub.shape[1] == 1
        if sub.shape[1] == 2: header = False
    return sub.to_csv(index=False, header=header).encode("utf-8"), header



# =====================================================================
# A4サマリーPDF（data_summary_a4 v7 ノートと同じ中身）の設定
# =====================================================================
# 【A4-1】UNIQUE_SHOW_MAX：基本情報の表に、ユニーク値を何個まで表示するか。例 5。空にはできません
UNIQUE_SHOW_MAX = 5
# 【A4-2】ROWS_PER_PAGE：基本情報の表で、A4 1ページに載せるカラム数。例 14。超えると自動で複数ページ
ROWS_PER_PAGE = 14
# 【A4-3】DISCRETE_MAX：ユニーク数がこの値以下の数値列は、ヒストグラムでなく棒グラフ（0/1フラグ・曜日コード向け）。例 10
DISCRETE_MAX = 10
# 【A4-4】CAT_TOP_N：文字列列の棒グラフ・色分けに出す上位カテゴリ数。例 8
CAT_TOP_N = 8
# 【A4-5】HIST_BINS：ヒストグラムの棒の本数。例 20
HIST_BINS = 20
# 【A4-6】GRAPH_PER_PAGE：グラフ一覧・目的変数との関係の A4 1ページのグラフ数。例 16
GRAPH_PER_PAGE = 16
# 【A4-7】CORR_METHOD：相関の種類。"pearson"（直線的な関係）か "spearman"（順位ベース・外れ値に強い）
CORR_METHOD = "pearson"
# 【A4-8】TARGET_MA：目的変数の推移に重ねる移動平均の点数。例 7（お弁当は平日だけなので5〜7）
TARGET_MA = 7
# 【A4-9】TARGET_GRAPH_PER_PAGE：「特徴量×目的変数（時系列）」の A4 1ページのグラフ数。例 9（3列×3段）
TARGET_GRAPH_PER_PAGE = 9
# 【A4-10】TIME_COL / TARGET：PDF作成時に、左の「📅日付列」「🎯目的変数」で自動的に上書きされます。
#   ""＝自動検出、"-"＝そのページを作らない。ここは触らなくてOK
TIME_COL = ""
TARGET = ""

import unicodedata
import warnings
import matplotlib.dates
from matplotlib.backends.backend_pdf import PdfPages

def _w(ch):
    """表示幅(全角=2, 半角=1)"""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def wrap(s, width=44, max_lines=3):
    """セル内テキストを表示幅(半角換算)で折り返し、行数オーバーは省略する"""
    lines = []
    for part in str(s).split("\n"):
        cur, cw = "", 0
        for ch in part:
            if cw + _w(ch) > width:
                lines.append(cur)
                cur, cw = "", 0
            cur += ch
            cw += _w(ch)
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "…"
    return "\n".join(lines)


def fmt(x):
    """数値を見やすい桁数に整形"""
    if pd.isna(x):
        return "-"
    if abs(x) >= 1000 or float(x).is_integer():
        return f"{x:,.0f}"
    return f"{x:.2f}"


def make_summary(df):
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):  # v7: 00:00:00 を省いて日付だけ表示
            s = s.dt.strftime("%Y-%m-%d") if (s.dropna().dt.normalize() == s.dropna()).all() else s.astype(str).where(s.notna())
        uniq = s.dropna().unique()
        nuniq = len(uniq)
        shown = ", ".join(map(str, uniq[:UNIQUE_SHOW_MAX]))
        uniq_txt = f"{nuniq}種\n{shown}" + (" …" if nuniq > UNIQUE_SHOW_MAX else "")
        nmiss = int(s.isna().sum())

        if pd.api.types.is_numeric_dtype(s) and nuniq > 0:
            d = s.describe()
            stat = (f"平均 {fmt(d['mean'])} / 標準偏差 {fmt(d['std'])}\n"
                    f"最小 {fmt(d['min'])} / 中央 {fmt(d['50%'])} / 最大 {fmt(d['max'])}")
        elif nuniq > 0:
            vc = s.value_counts()
            stat = f"最頻値 {vc.index[0]}({vc.iloc[0]}件)"
        else:
            stat = "全て欠損"

        rows.append({
            "カラム名": c,
            "データ型": str(s.dtype),
            "欠損": f"{nmiss}\n({nmiss / n:.0%})",
            "ユニーク値": uniq_txt,
            "基本統計": stat,
        })
    return pd.DataFrame(rows)


def draw_pages(summary, title, shape, pdf, png_prefix=None):
    """summaryをA4横のページに分けて描画"""
    widths = [0.14, 0.09, 0.07, 0.32, 0.38]
    wrap_w = [24, 16, 10, 54, 66]  # 各列の折り返し幅(半角換算。全角1文字=2)
    pages = [summary.iloc[i:i + ROWS_PER_PAGE] for i in range(0, len(summary), ROWS_PER_PAGE)]
    for pi, part in enumerate(pages, 1):
        fig = plt.figure(figsize=(11.69, 8.27))  # A4横
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.86])
        ax.axis("off")
        fig.text(0.03, 0.935, f"{title}  ({shape[0]:,}行 × {shape[1]}列)", fontsize=16, weight="bold")
        fig.text(0.97, 0.94, f"{pi}/{len(pages)}", fontsize=10, ha="right")
        cell_text = [[wrap(v, w) for v, w in zip(r, wrap_w)] for r in part.values]
        tbl = ax.table(cellText=cell_text, colLabels=part.columns, colWidths=widths,
                       loc="upper center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        row_h = 0.92 / ROWS_PER_PAGE
        for (r, c), cell in tbl.get_celld().items():
            cell.set_height(0.04 if r == 0 else row_h)
            cell.set_edgecolor("#999999")
            if r == 0:
                cell.set_facecolor("#0b1d3a")
                cell.get_text().set_color("#d4af37")
                cell.get_text().set_weight("bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f1f3f7")
        pdf.savefig(fig)
        if png_prefix:
            fig.savefig(f"{png_prefix}_p{pi}.png", dpi=150)
        plt.close(fig)



def find_time_col(df):
    """日付列を決める(TIME_COL指定 or 自動検出)。なければ None"""
    if TIME_COL == "-":
        return None
    if TIME_COL:
        return TIME_COL if TIME_COL in df.columns else None
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = pd.to_datetime(df[c], errors="coerce")
        if t.notna().mean() >= 0.9:
            return c
    return None


def is_discrete(s):
    return (not pd.api.types.is_numeric_dtype(s)) or s.nunique() <= DISCRETE_MAX


def style_ax(ax):
    ax.tick_params(labelsize=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_bar(ax, s, name):
    vc = s.value_counts().head(CAT_TOP_N)
    vc = vc.sort_index() if pd.api.types.is_numeric_dtype(s) else vc.iloc[::-1]
    labels = [str(i)[:6] for i in vc.index]
    if pd.api.types.is_numeric_dtype(s):
        ax.bar(labels, vc.values, color="#0b1d3a")
    else:
        ax.barh(labels, vc.values, color="#0b1d3a")
    more = s.nunique() - len(vc)
    ax.set_title(f"{name}(件数)" + (f" 他{more}種" if more > 0 else ""), fontsize=8, loc="left")
    style_ax(ax)


def draw_numeric(fig, spec, s, name):
    inner = spec.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.45)
    ax1 = fig.add_subplot(inner[0])
    ax2 = fig.add_subplot(inner[1])
    v = s.dropna()
    ax1.hist(v, bins=HIST_BINS, color="#0b1d3a", edgecolor="white", linewidth=0.3)
    ax1.set_title(f"{name}(ヒストグラム)", fontsize=8, loc="left")
    ax1.set_ylabel("度数", fontsize=6)
    ax2.boxplot(v, vert=False, widths=0.6, patch_artist=True,
                boxprops=dict(facecolor="#d4af37", edgecolor="#0b1d3a"),
                medianprops=dict(color="#0b1d3a"), flierprops=dict(markersize=2))
    ax2.set_yticks([])
    ax2.set_xlabel("箱ひげ図", fontsize=6)
    style_ax(ax1)
    style_ax(ax2)


def draw_line(ax, t, s, name):
    d = pd.DataFrame({"t": t, "v": s}).dropna().sort_values("t")
    ax.plot(d["t"], d["v"], color="#0b1d3a", linewidth=0.8)
    ax.set_title(f"{name}(時系列)", fontsize=8, loc="left")
    ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(maxticks=5))
    ax.tick_params(axis="x", rotation=30)
    style_ax(ax)


def draw_graph_pages(df, title, pdf, png_prefix=None):
    """全カラムのグラフ(+時系列)を、1枚にできるだけ詰めて描画"""
    tcol = find_time_col(df)
    items = []  # (種類, 列名)
    for c in df.columns:
        if c == tcol or df[c].dropna().empty:
            continue
        items.append(("bar" if is_discrete(df[c]) else "num", c))
    t = None
    if tcol:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = pd.to_datetime(df[tcol], errors="coerce")
        items += [("line", c) for c in df.columns
                  if c != tcol and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > DISCRETE_MAX]

    label = "グラフ一覧" + (f"(時系列の横軸:{tcol})" if tcol else "")
    pages = [items[i:i + GRAPH_PER_PAGE] for i in range(0, len(items), GRAPH_PER_PAGE)]
    for pi, part in enumerate(pages, 1):
        n = len(part)
        ncol = int(np.ceil(np.sqrt(n * 1.45)))
        nrow = int(np.ceil(n / ncol))
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.03, 0.95, f"{title}  {label}", fontsize=14, weight="bold")
        fig.text(0.97, 0.95, f"{pi}/{len(pages)}", fontsize=10, ha="right")
        gs = fig.add_gridspec(nrow, ncol, left=0.08, right=0.98, top=0.89, bottom=0.07,
                              wspace=0.75, hspace=0.6)
        for k, (kind, c) in enumerate(part):
            spec = gs[k // ncol, k % ncol]
            if kind == "num":
                draw_numeric(fig, spec, df[c], c)
            elif kind == "bar":
                draw_bar(fig.add_subplot(spec), df[c], c)
            else:
                draw_line(fig.add_subplot(spec), t, df[c], c)
        pdf.savefig(fig)
        if png_prefix:
            fig.savefig(f"{png_prefix}_グラフ一覧_p{pi}.png", dpi=150)
        plt.close(fig)

def draw_corr_page(df, title, pdf, png_prefix=None):
    num = df.select_dtypes("number")
    num = num.loc[:, num.nunique() > 1]  # 定数列・全欠損列は除外
    if num.shape[1] < 2:
        print(f"[{title}] 数値列が2つ未満のため相関ヒートマップはスキップ")
        return
    corr = num.corr(method=CORR_METHOD)
    n = corr.shape[0]
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.03, 0.95, f"{title}  相関係数ヒートマップ({CORR_METHOD})", fontsize=16, weight="bold")
    ax = fig.add_axes([0.14, 0.12, 0.70, 0.74])
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("相関係数(-1〜+1)", fontsize=9)
    pdf.savefig(fig)
    if png_prefix:
        fig.savefig(f"{png_prefix}_相関ヒートマップ.png", dpi=150)
    plt.close(fig)

def find_target(df):
    """目的変数を決める(TARGET指定 or 候補名から自動)。数値が2行以上ない列は使わない。なければ None"""
    if TARGET == "-":
        return None
    def ok(c):
        return c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 2
    if TARGET:
        return TARGET if ok(TARGET) else None
    return next((c for c in TARGET_CANDIDATES if ok(c)), None)


def _feature_kind(s):
    """flag(0/1) / disc(種類が少ない) / num(連続値) / None(種類が多すぎる文字列などは描かない)"""
    v = s.dropna()
    if v.empty:
        return None
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        if set(pd.to_numeric(v, errors="coerce").unique()).issubset({0, 1}):
            return "flag"
        return "disc" if v.nunique() <= DISCRETE_MAX else "num"
    return "disc" if v.nunique() <= CAT_TOP_N else None


def _shade_test(ax, test_range):
    if test_range is not None:
        ax.axvspan(test_range[0], test_range[1], color="#d4af37", alpha=0.15, lw=0)


def draw_target_overview(df, title, pdf, png_prefix=None, test_range=None):
    """④-1 目的変数の推移(移動平均・test期間) + 曜日別・月別の平均"""
    tcol, ycol = find_time_col(df), find_target(df)
    if not tcol or not ycol:
        print(f"[{title}] 日付列または目的変数がないため「目的変数×時系列」はスキップ"
              f"(日付列={tcol} / 目的変数={ycol})")
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = pd.to_datetime(df[tcol], errors="coerce")
    d = pd.DataFrame({"t": t, "y": pd.to_numeric(df[ycol], errors="coerce")}).dropna().sort_values("t")
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.03, 0.95, f"{title}  目的変数「{ycol}」の時系列(横軸:{tcol})", fontsize=14, weight="bold")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.89, bottom=0.08, hspace=0.45, wspace=0.25,
                          height_ratios=[1.4, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(d["t"], d["y"], color="#0b1d3a", lw=0.8, marker="o", ms=2, label=ycol)
    ma = d["y"].rolling(TARGET_MA, min_periods=1).mean()
    ax.plot(d["t"], ma, color="#d4af37", lw=2, label=f"{TARGET_MA}点移動平均")
    _shade_test(ax, test_range)
    if test_range is not None:
        ax.text(test_range[0], ax.get_ylim()[1], " test期間", fontsize=8, va="top", color="#8a6d10")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"{ycol} の推移  (平均 {fmt(d['y'].mean())} / 最初の{TARGET_MA}点平均 {fmt(d['y'].head(TARGET_MA).mean())}"
                 f" → 最後の{TARGET_MA}点平均 {fmt(d['y'].tail(TARGET_MA).mean())})", fontsize=9, loc="left")
    ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(maxticks=10))
    style_ax(ax)
    # 曜日別・月別の平均(棒の上に件数)
    for k, (lab, key, names) in enumerate([("曜日別", d["t"].dt.dayofweek, list("月火水木金土日")),
                                           ("月別", d["t"].dt.month, [f"{i}月" for i in range(1, 13)])]):
        a = fig.add_subplot(gs[1, k])
        g = d.groupby(key)["y"].agg(["mean", "count"])
        a.bar([names[i] if lab == "曜日別" else names[i - 1] for i in g.index], g["mean"], color="#0b1d3a")
        for x, (m_, n_) in enumerate(zip(g["mean"], g["count"])):
            a.text(x, m_, f"{n_}件", ha="center", va="bottom", fontsize=6)
        a.set_title(f"{lab}の{ycol}平均", fontsize=9, loc="left")
        style_ax(a)
    pdf.savefig(fig)
    if png_prefix:
        fig.savefig(f"{png_prefix}_目的変数時系列.png", dpi=150)
    plt.close(fig)
    return True


def _draw_feature_time(ax, t, y, s, c, ycol, kind):
    """1つの特徴量と目的変数を、同じ時間軸に重ねる"""
    d = pd.DataFrame({"t": t, "y": y, "f": s}).dropna(subset=["t"]).sort_values("t")
    ax.plot(d["t"], d["y"], color="#9aa3b5", lw=0.7, zorder=1)
    if kind == "num":
        f = pd.to_numeric(d["f"], errors="coerce")
        ax2 = ax.twinx()
        ax2.plot(d["t"], f, color="#d4af37", lw=0.9)
        ax2.tick_params(labelsize=5, colors="#8a6d10")
        for sp in ("top",):
            ax2.spines[sp].set_visible(False)
        r = pd.concat([d["y"], f], axis=1).dropna().corr().iloc[0, 1] if d["y"].notna().sum() > 2 else np.nan
        ax.set_title(f"{c}(金)× {ycol}(灰)  r={r:.2f}" if pd.notna(r) else f"{c}(金)× {ycol}(灰)", fontsize=7, loc="left")
    elif kind == "flag":
        f = pd.to_numeric(d["f"], errors="coerce")
        on = d[f == 1]
        ax.scatter(on["t"], on["y"], color="#c0392b", s=9, zorder=3)
        m1, m0 = d.loc[f == 1, "y"].mean(), d.loc[f == 0, "y"].mean()
        ax.set_title(f"{c}=1(赤 {int((f == 1).sum())}件)  平均 {fmt(m1)} vs 0のとき {fmt(m0)}", fontsize=7, loc="left")
    else:
        top = d["f"].value_counts().head(CAT_TOP_N).index
        cmap = plt.get_cmap("tab10")
        for i, v in enumerate(top):
            p = d[d["f"] == v]
            ax.scatter(p["t"], p["y"], s=7, color=cmap(i % 10), label=str(v)[:6], zorder=3)
        ax.legend(fontsize=5, ncol=2, loc="upper right", frameon=False, markerscale=0.8)
        ax.set_title(f"{c} 別の {ycol}", fontsize=7, loc="left")
    ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(maxticks=4))
    ax.tick_params(axis="x", rotation=30)
    style_ax(ax)


def draw_target_feature_pages(df, title, pdf, png_prefix=None, test_range=None):
    """④-2 各特徴量 × 目的変数 を同じ時間軸で(連続値=2軸の折れ線 / フラグ=1の日に赤点 / カテゴリ=色分け)"""
    tcol, ycol = find_time_col(df), find_target(df)
    if not tcol or not ycol:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = pd.to_datetime(df[tcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    has_y = y.notna()
    items = [(c, _feature_kind(df.loc[has_y, c])) for c in df.columns if c not in (tcol, ycol, "_set")]
    items = [(c, k) for c, k in items if k]
    if not items:
        return
    pages = [items[i:i + TARGET_GRAPH_PER_PAGE] for i in range(0, len(items), TARGET_GRAPH_PER_PAGE)]
    for pi, part in enumerate(pages, 1):
        n = len(part); ncol = 3 if n > 4 else 2; nrow = int(np.ceil(n / ncol))
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.03, 0.95, f"{title}  特徴量 × 目的変数「{ycol}」(時系列)", fontsize=14, weight="bold")
        fig.text(0.97, 0.95, f"{pi}/{len(pages)}", fontsize=10, ha="right")
        gs = fig.add_gridspec(max(nrow, 2), ncol, left=0.06, right=0.95, top=0.89, bottom=0.07, wspace=0.35, hspace=0.75)
        for k, (c, kind) in enumerate(part):
            ax = fig.add_subplot(gs[k // ncol, k % ncol])
            _draw_feature_time(ax, t[has_y], y[has_y], df.loc[has_y, c], c, ycol, kind)
        pdf.savefig(fig)
        if png_prefix:
            fig.savefig(f"{png_prefix}_特徴量x目的変数_p{pi}.png", dpi=150)
        plt.close(fig)


def draw_target_relation_pages(df, title, pdf, png_prefix=None):
    """⑤ 目的変数との関係(日付列がなくても作る):連続値=散布図 / フラグ・カテゴリ=箱ひげ図"""
    ycol = find_target(df)
    if not ycol:
        return
    y = pd.to_numeric(df[ycol], errors="coerce")
    has_y = y.notna()
    tcol = find_time_col(df)
    items = [(c, _feature_kind(df.loc[has_y, c])) for c in df.columns if c not in (tcol, ycol, "_set")]
    items = [(c, k) for c, k in items if k]
    if not items:
        return
    pages = [items[i:i + GRAPH_PER_PAGE] for i in range(0, len(items), GRAPH_PER_PAGE)]
    for pi, part in enumerate(pages, 1):
        n = len(part); ncol = int(np.ceil(np.sqrt(n * 1.45))); nrow = int(np.ceil(n / ncol))
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.03, 0.95, f"{title}  目的変数「{ycol}」との関係(散布図・箱ひげ)", fontsize=14, weight="bold")
        fig.text(0.97, 0.95, f"{pi}/{len(pages)}", fontsize=10, ha="right")
        gs = fig.add_gridspec(nrow, ncol, left=0.06, right=0.98, top=0.89, bottom=0.07, wspace=0.5, hspace=0.7)
        for k, (c, kind) in enumerate(part):
            ax = fig.add_subplot(gs[k // ncol, k % ncol])
            s = df.loc[has_y, c]; yy = y[has_y]
            if kind == "num":
                f = pd.to_numeric(s, errors="coerce")
                ax.scatter(f, yy, s=5, color="#0b1d3a", alpha=0.6)
                r = pd.concat([f, yy], axis=1).dropna().corr().iloc[0, 1]
                ax.set_title(f"{c}  r={r:.2f}", fontsize=7, loc="left")
            else:
                top = s.value_counts().head(CAT_TOP_N).index
                top = sorted(top) if kind == "flag" or pd.api.types.is_numeric_dtype(s) else list(top)
                data = [yy[s == v].dropna() for v in top]
                bp = ax.boxplot(data, widths=0.6, patch_artist=True,
                                boxprops=dict(facecolor="#d4af37", edgecolor="#0b1d3a"),
                                medianprops=dict(color="#0b1d3a"), flierprops=dict(markersize=2))
                ax.set_xticks(range(1, len(top) + 1))
                ax.set_xticklabels([f"{str(v)[:5]}\n{len(x)}件" for v, x in zip(top, data)], fontsize=5)
                ax.set_title(f"{c}", fontsize=7, loc="left")
            style_ax(ax)
        pdf.savefig(fig)
        if png_prefix:
            fig.savefig(f"{png_prefix}_目的変数との関係_p{pi}.png", dpi=150)
        plt.close(fig)



def make_a4_pdf(frames, target_name, time_name, include_combined=False):
    """frames: {"train": df, "test": df} → {ファイル名: PDFのバイト列}。ノートv7の「実行」セルと同じ流れ"""
    global TIME_COL, TARGET
    TIME_COL = time_name or ""
    TARGET = target_name or "-"
    test_range = None
    if "test" in frames:
        tc = find_time_col(frames["test"])
        if tc:
            tt = pd.to_datetime(frames["test"][tc], errors="coerce").dropna()
            if len(tt): test_range = (tt.min(), tt.max())
    jobs = [(n, d, n) for n, d in frames.items()]
    if include_combined and len(frames) == 2:
        jobs.append(("all", pd.concat([frames["train"].assign(_set="train"), frames["test"].assign(_set="test")], ignore_index=True), "train+test合体"))
    out = {}
    for name, df, label in jobs:
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            draw_pages(make_summary(df), f"{label} 基本データサマリー", df.shape, pdf)
            draw_graph_pages(df, label, pdf)
            draw_corr_page(df, label, pdf)
            if find_target(df):
                tr = test_range if name != "test" else None
                if draw_target_overview(df, label, pdf, None, tr):
                    draw_target_feature_pages(df, label, pdf, None, tr)
                draw_target_relation_pages(df, label, pdf)
        out[f"data_summary_{name}.pdf"] = buf.getvalue()
        plt.close("all")
    return out

# =====================================================================
# メインUI
# =====================================================================
st.title("🚀 直感型 データ前処理・特徴量生成アプリ v5")

defaults = {"df": None, "processed_df": None, "logs": [], "original_size": None, "current_file_id": None,
            "sample_raw": None, "sample_name": None, "orig_test_rows": 0, "ml_result": None, "ml_history": [], "a4_pdfs": None, "data_name": "未分類", "ml_records": []}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v


def set_loaded(df, fid, size):
    st.session_state.df = fix_dataframe_for_streamlit(df)
    st.session_state.original_size = size; st.session_state.current_file_id = fid
    st.session_state.processed_df = None; st.session_state.logs = []; st.session_state.ml_result = None
    st.session_state.orig_test_rows = int((df["_source_"] == "test").sum())


col_ctrl, col_view = st.columns([4, 6], gap="large")

with col_ctrl:
    c_head1, c_head2 = st.columns([6, 4])
    with c_head1: st.header("⚙️ 設定と実行")
    with c_head2:
        if st.session_state.df is not None and st.button("🔄 最初からやり直す", use_container_width=True):
            for k, v in defaults.items():
                if k not in ("ml_history", "ml_records"): st.session_state[k] = v
            st.rerun()

    with st.expander("📂 0. データ読み込み", expanded=(st.session_state.df is None)):
        read_mode = st.radio("読み込みモード", ["単一ファイル", "Train / Test 個別", "Train / Test まとめてZIP"], horizontal=True)
        if read_mode == "単一ファイル":
            f = st.file_uploader("ファイルをアップロード")
            if f:
                fid = f"single_{f.name}_{f.size}"
                if st.session_state.current_file_id != fid:
                    d = read_file_safely(f, f.name); d["_source_"] = "single"; set_loaded(d, fid, f.size); st.rerun()
        elif read_mode == "Train / Test 個別":
            c1, c2 = st.columns(2)
            with c1: train_file = st.file_uploader("Train")
            with c2: test_file = st.file_uploader("Test")
            if train_file and test_file:
                fid = f"multi_{train_file.name}_{train_file.size}_{test_file.name}_{test_file.size}"
                if st.session_state.current_file_id != fid:
                    tr = read_file_safely(train_file, train_file.name); te = read_file_safely(test_file, test_file.name)
                    tr["_source_"] = "train"; te["_source_"] = "test"
                    set_loaded(pd.concat([tr, te], axis=0, ignore_index=True), fid, train_file.size + test_file.size); st.rerun()
        else:
            uz = st.file_uploader("ZIPファイルをアップロード (.zip)")
            if uz:
                fid = f"zip_{uz.name}_{uz.size}"
                if st.session_state.current_file_id != fid:
                    try:
                        with zipfile.ZipFile(uz, 'r') as z:
                            files = [n for n in z.namelist() if not n.endswith('/') and '__MACOSX' not in n
                                     and not n.split('/')[-1].startswith('.')]
                            base = lambda n: n.split('/')[-1].lower()
                            sample_name = next((n for n in files if 'sample' in base(n) or 'submission' in base(n)), None)
                            rest = [n for n in files if n != sample_name]
                            train_name = next((n for n in rest if 'train' in base(n)), None)
                            test_name = next((n for n in rest if 'test' in base(n) and n != train_name), None)
                            if train_name and test_name:
                                tr = read_file_safely(io.BytesIO(z.read(train_name)), train_name)
                                te = read_file_safely(io.BytesIO(z.read(test_name)), test_name)
                                tr["_source_"] = "train"; te["_source_"] = "test"
                                if sample_name:
                                    st.session_state.sample_raw = z.read(sample_name); st.session_state.sample_name = base(sample_name)
                                set_loaded(pd.concat([tr, te], axis=0, ignore_index=True), fid, uz.size); st.rerun()
                            else:
                                st.error(f"ZIPの中に 'train' と 'test' を含むファイルが見つかりませんでした。中身: {files}")
                    except zipfile.BadZipFile as e:
                        st.error(f"ZIPファイルの読み込みに失敗しました: {e}")
        st.markdown("**提出見本（sample）※任意**：🤖機械学習タブで提出CSVを見本と同じ形にします")
        sf = st.file_uploader("提出見本 (sample.csv)", key="sample_up")
        if sf and st.session_state.sample_name != sf.name:
            st.session_state.sample_raw = sf.getvalue(); st.session_state.sample_name = sf.name
        if st.session_state.sample_name: st.caption(f"✅ 見本: {st.session_state.sample_name}")

    if st.session_state.df is not None:
        base_df = st.session_state.df
        df_cols = [c for c in base_df.columns.tolist() if c != "_source_"]
        current_num_cols = get_numeric_columns(base_df)
        default_target = next((c for c in TARGET_CANDIDATES if c in df_cols), None)
        default_date = next((c for c in DATE_CANDIDATES if c in df_cols), None) or next((c for c in df_cols if is_date_like(base_df[c])), None)

        st.write("---")
        st.subheader("🛠 前処理ワークフロー")
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            target_col = st.selectbox("🎯 目的変数（削除・補完から守る列）", ["(なし)"] + df_cols,
                                      index=df_cols.index(default_target) + 1 if default_target else 0)
        with c_t2:
            date_col_sel = st.selectbox("📅 日付列（ラグ・過去平均・機械学習の並び順）", ["(なし)"] + df_cols,
                                        index=df_cols.index(default_date) + 1 if default_date else 0)
        target = target_col if target_col != "(なし)" else None
        date_col = date_col_sel if date_col_sel != "(なし)" else None
        protect_test = st.checkbox("🛡 行を消す処理で test の行は消さない（推奨：提出行数がずれない）", value=True)

        safe_df_cols = [c for c in df_cols if c != target_col]
        safe_num_cols = [c for c in current_num_cols if c != target_col]
        continuous_num_cols = [c for c in safe_num_cols if not str(c).endswith('_ID')]

        with st.expander("🪄 STEP1: 複雑な文字列の処理", expanded=True):
            tab1, tab2 = st.tabs(["✂️ 複数列に分割", "🔄 カテゴリ処理 (型変換/ID化/ダミー化)"])
            with tab1:
                c1, c2 = st.columns([5, 5])
                with c1:
                    split_col = st.selectbox("分割する列", ["(なし)"] + safe_df_cols, key="split_col_step1")
                    split_strategy = st.radio("方式", ["単語ごとに専用列を作成", "左詰めで分割 (最大N列指定)"])
                    split_max = st.number_input("最大列数", min_value=2, value=5) if split_strategy.startswith("左詰め") else 5
                with c2:
                    split_delim = st.text_input("区切り文字", value="自動認識 (/, |, [], 空白, 全角半角)", key="split_delim")
                    split_drop = st.checkbox("処理後に元の列を削除", value=True)
            with tab2:
                st.caption("💡 「すべて自動」は、日付らしい列を自動で対象外にします（日付が消えないように）")
                st.markdown("##### 🅰️ カテゴリ型 (category) への一括変換 (LightGBM推奨)")
                cat_auto = st.checkbox("🪄 すべての文字カラムを category 型に変換", value=False)
                cat_columns = [] if cat_auto else st.multiselect("category型にする列", safe_df_cols)
                st.divider()
                st.markdown("##### 🔢 カテゴリの一括数値ID化 (Label Encoding)")
                le_auto = st.checkbox("🪄 すべての文字カラムを全自動で数値IDに変換", value=False)
                le_columns = [] if le_auto else st.multiselect("ID化する列", safe_df_cols)
                le_drop = st.checkbox("元の文字カラムを自動で削除", value=True, key="le_drop")
                st.divider()
                st.markdown("##### 🔄 ダミー化 (One-Hot Encoding)")
                dummy_auto = st.checkbox("🪄 条件に合う文字カラムを全自動で一括ダミー化する", value=False)
                dummy_max_missing, dummy_min_freq, dummy_max_unique = 50, 1.0, AUTO_DUMMY_MAX_UNIQUE
                if dummy_auto:
                    c_d1, c_d2, c_d3 = st.columns(3)
                    with c_d1: dummy_max_missing = st.number_input("欠損率がこれより多い列は対象外 (%)", 0, 100, 50)
                    with c_d2: dummy_max_unique = st.number_input("種類がこれより多い列は対象外", 1, 10000, AUTO_DUMMY_MAX_UNIQUE)
                    with c_d3: dummy_min_freq = st.number_input("出現率がこれ未満のフラグを削除 (%)", 0.0, 100.0, 1.0, step=0.1)
                onehot_columns = [] if dummy_auto else st.multiselect("ダミー化する列を手動で選択", safe_df_cols)
                st.divider()
                st.markdown("##### 🔄 複数キーワードの一括フラグ化")
                c1_2, c2_2 = st.columns([6, 4])
                with c1_2: ml_column = st.selectbox("複数キーワードが入った列", ["(なし)"] + safe_df_cols, key="ml_col")
                with c2_2: ml_delim = st.text_input("区切り文字", value="／", key="ml_delim")
                ml_drop = st.checkbox("処理後に元の列を自動で削除", value=True, key="ml_drop")

        with st.expander("🛠 STEP2: 基本処理 (欠損値・型変換など)"):
            st.caption("※ 型変換はすべての処理より先に実行されます")
            c_type1, c_type2 = st.columns(2)
            with c_type1: type_conv_cols = st.multiselect("型を変換する列を選択", safe_df_cols)
            with c_type2: type_conv_target = st.selectbox("変換後のデータ型", ["文字列 (str)", "整数 (int)", "小数 (float)", "日付 (datetime)"])
            st.divider()
            missing_action = st.radio("欠損値", ["変更しない", "数値列を0で補完", "数値列を中央値で補完(trainの中央値)",
                                              "目的変数が欠損の行を削除(trainのみ)", "どこかに欠損がある行を削除(trainのみ)"])
            fill_text_missing = st.checkbox("文字列の列の欠損を「不明」で埋める", value=False)
            outlier_action = st.radio("異常値", ["変更しない", "IQRで異常値行を削除"], horizontal=True)
            iqr_cols, iqr_k = [], 1.5
            if outlier_action != "変更しない":
                st.caption("対象はtrainの行だけ。空なら連続値の列（0/1フラグ・ID・種類10以下の列は除く）を自動で判定")
                ci1, ci2 = st.columns([7, 3])
                with ci1: iqr_cols = st.multiselect("IQRで判定する列（空なら自動）", continuous_num_cols)
                with ci2: iqr_k = st.number_input("倍率（1.5が標準）", 0.5, 10.0, 1.5, step=0.5)
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
                    "【変換】数値を任意の幅でグループ化 (例: 10刻みで「〇〇代」)",
                    "【変換】カスタム境界値で数値をカテゴリ化",
                    "【数値計算】2つの列で四則演算 (+, -, *, /)",
                    "【数値計算】対数変換 (Log1p - 歪み補正)",
                    "【変換】数値のビン分割 (10分割)",
                    "【時系列】カテゴリ別の過去実績(y)の平均 (ターゲットエンコーディング)",
                    "【時系列】過去の値を取得 (ラグ特徴量)",
                    "【時系列】日付列から年・月・日・曜日を抽出",
                    "【時系列】日付のフォーマット変換",
                    "【抽出】「徒歩〇分」から最短分数を抽出",
                    "【抽出】「築〇年」から年数を抽出 (単一の数値に変換)",
                    "【抽出】「〇年〇ヶ月」を年と月に別々の列に分割",
                    "【抽出】最初の数値を抽出",
                    "【テキスト】文字数カウント",
                    "【フラグ】文字が入力されていれば1、空欄なら0 (有無フラグ)",
                    "【フラグ】キーワードが含まれていれば1、なければ0",
                    "【フラグ】複数ルールの専用フラグを一括生成 (テキスト入力)"
                ])
            with c1:
                if fe_method in ["【自動生成】数値カラム同士の総当たり割り算 (A ÷ B)", "【数値計算】2つの列で四則演算 (+, -, *, /)"]: allowed = continuous_num_cols
                elif fe_method == "【時系列】過去の値を取得 (ラグ特徴量)": allowed = df_cols
                else: allowed = safe_df_cols
                fe_column = st.selectbox("基準となる列（元データ）", ["(なし)"] + allowed)
            c3, c4 = st.columns([4, 6])
            with c3: fe_prefix = st.text_input("追加する特徴量の接頭辞 (任意)")
            with c4:
                date_format = "yyyy-MM-dd"; fe_keyword = ""; fe_col2 = None; fe_op = "+"; fe_custom_bins = ""; fe_dict_text = ""
                fe_num_col = None; fe_aggs = ["mean"]; fe_group_width = 10; fe_group_suffix = "代"
                fe_lag_n = 1; fe_lag_group = "(なし)"; fe_auto_cols = []; fe_flag_suffix = ""
                if fe_method == "【時系列】日付のフォーマット変換":
                    date_format = st.selectbox("フォーマット", ["yyyy-m-d", "yyyy-MM-dd", "yyyy/m/d", "yyyy/MM/dd", "yyyyMMdd"])
                elif fe_method == "【自動生成】数値カラム同士の総当たり割り算 (A ÷ B)":
                    fe_auto_cols = st.multiselect("対象", continuous_num_cols, default=continuous_num_cols[:min(5, len(continuous_num_cols))])
                elif fe_method == "【時系列】カテゴリ別の過去実績(y)の平均 (ターゲットエンコーディング)":
                    fe_lag_n = st.number_input("何回分の過去平均を取るか", min_value=1, value=7)
                    st.caption("📅 日付列を選んでいれば日付順で計算。testの行には直前の平均を引き継ぎます")
                elif fe_method == "【時系列】過去の値を取得 (ラグ特徴量)":
                    col4_1, col4_2 = st.columns(2)
                    with col4_1: fe_lag_n = st.number_input("何行前", min_value=1, value=1)
                    with col4_2: fe_lag_group = st.selectbox("グループ列", ["(なし)"] + safe_df_cols)
                    st.caption("⚠ 目的変数のラグは、testではN行先までしか値が入りません")
                elif fe_method == "【変換】数値を任意の幅でグループ化 (例: 10刻みで「〇〇代」)":
                    col4_1, col4_2 = st.columns(2)
                    with col4_1: fe_group_width = st.number_input("区切り幅", min_value=1, value=10)
                    with col4_2: fe_group_suffix = st.text_input("後ろにつける文字", value="代")
                elif fe_method == "【変換】カスタム境界値で数値をカテゴリ化":
                    fe_custom_bins = st.text_input("境界値を入力 (カンマ区切り)", value="200, 300, 400, 500")
                elif fe_method == "【フラグ】キーワードが含まれていれば1、なければ0":
                    c_f1, c_f2 = st.columns([4, 6])
                    with c_f1: fe_flag_suffix = st.text_input("作成する列名(後ろにつく名前)", value="フラグ")
                    with c_f2: fe_keyword = st.text_input("キーワード (カンマ区切り)", placeholder="例: 肉, 豚, 牛")
                elif fe_method == "【フラグ】複数ルールの専用フラグを一括生成 (テキスト入力)":
                    st.info("💡 「カラム名: キーワード1, キーワード2...」の形式。複数ヒットしてもフラグは 1")
                    default_dict = "curry: カレー\nfish: 魚, 鮭, 鯖, さば, サバ, ぶり, 鰆, さわら, イカ, 海老, エビ, えび, カキ, あじ, アジ, タラ, 鯛\nfry: フライ, カツ, 唐揚, から揚, 天ぷら\nmeat: 肉, 豚, 鶏, チキン, 牛, ハンバーグ, ロース"
                    fe_dict_text = st.text_area("生成ルール (複数行入力可)", value=default_dict, height=120)
                elif fe_method == "【数値計算】2つの列で四則演算 (+, -, *, /)":
                    col4_1, col4_2 = st.columns([3, 7])
                    with col4_1: fe_op = st.radio("計算", ["+", "-", "×", "÷"], horizontal=True)
                    with col4_2: fe_col2 = st.selectbox("2つ目の列", continuous_num_cols)
                elif fe_method == "【集約】カテゴリごとの平均値などを算出 (複数選択可)":
                    col4_1, col4_2 = st.columns(2)
                    with col4_1: fe_num_col = st.selectbox("数値列", ["(なし)"] + continuous_num_cols)
                    with col4_2: fe_aggs = st.multiselect("方法", ["mean", "median", "sum", "std", "max", "min", "count"], default=["mean"])
                    st.caption("集計は train の行だけで計算します（testの値は使わない）")

        with st.expander("🗑 STEP4: 大掃除 (行・列の削除)"):
            drop_by_name = st.text_input("列名にこの文字が含まれていたら削除 (カンマ区切り)")
            min_count = st.number_input("trainで「1」がこの件数以下の0/1列を削除（0なら何もしない）", min_value=0, value=0)
            drop_columns = st.multiselect("手動で選んで削除", safe_df_cols)
            st.divider()
            st.markdown("##### 🎯 行の削除 (フィルタリング)　※🛡がONならtestの行は消しません")
            c_f1, c_f2 = st.columns(2)
            with c_f1: range_column = st.selectbox("数値フィルタ列 (指定範囲外を削除)", ["(なし)"] + continuous_num_cols)
            with c_f2:
                min_value = st.text_input("残す最小値")
                max_value = st.text_input("残す最大値")
            c_d1, c_d2, c_d3 = st.columns([4, 3, 3])
            with c_d1: date_filter_col = st.selectbox("日付フィルタ列", ["(なし)"] + df_cols)
            with c_d2: date_min = st.text_input("開始日 (例: 2013-11-01)")
            with c_d3: date_max = st.text_input("終了日 (例: 2014-05-31)")
            st.divider()
            c_c1, c_c2 = st.columns([4, 6])
            with c_c1: cat_filter_col = st.selectbox("カテゴリ・文字フィルタ列", ["(なし)"] + safe_df_cols)
            with c_c2:
                cat_filter_vals = []
                if cat_filter_col != "(なし)":
                    vc = base_df[cat_filter_col].value_counts(dropna=False)
                    val_map = {f"{v} ({n}件)": v for v, n in vc.items()}
                    cat_filter_vals = [val_map[o] for o in st.multiselect("削除したい値を選択", list(val_map.keys()))]

        if st.button("🚀 前処理を実行する", type="primary", use_container_width=True):
            P = dict(target=target, date_col=date_col, protect_test=protect_test, clean_text=clean_text,
                     type_conv_cols=type_conv_cols, type_conv_target=type_conv_target,
                     split_col=split_col, split_strategy=split_strategy, split_max=split_max, split_delim=split_delim, split_drop=split_drop,
                     ml_column=ml_column, ml_delim=ml_delim, ml_drop=ml_drop,
                     dummy_auto=dummy_auto, dummy_max_missing=dummy_max_missing, dummy_max_unique=dummy_max_unique,
                     dummy_min_freq=dummy_min_freq, onehot_columns=onehot_columns,
                     cat_auto=cat_auto, cat_columns=cat_columns, le_auto=le_auto, le_columns=le_columns, le_drop=le_drop,
                     missing_action=missing_action, fill_text_missing=fill_text_missing, duplicate=duplicate,
                     outlier_action=outlier_action, iqr_cols=iqr_cols, iqr_k=iqr_k,
                     fe_method=fe_method, fe_column=fe_column, fe_prefix=fe_prefix, date_format=date_format, fe_keyword=fe_keyword,
                     fe_col2=fe_col2, fe_op=fe_op, fe_custom_bins=fe_custom_bins, fe_dict_text=fe_dict_text, fe_num_col=fe_num_col,
                     fe_aggs=fe_aggs, fe_group_width=fe_group_width, fe_group_suffix=fe_group_suffix, fe_lag_n=fe_lag_n,
                     fe_lag_group=fe_lag_group, fe_auto_cols=fe_auto_cols, fe_flag_suffix=fe_flag_suffix,
                     drop_by_name=drop_by_name, min_count=min_count, drop_columns=drop_columns,
                     range_column=range_column, min_value=min_value, max_value=max_value,
                     date_filter_col=date_filter_col, date_min=date_min, date_max=date_max,
                     cat_filter_col=cat_filter_col, cat_filter_vals=cat_filter_vals)
            with st.spinner("処理中..."):
                try:
                    out, log = run_preprocess(base_df, P)
                    st.session_state.processed_df = out; st.session_state.logs = log; st.session_state.ml_result = None; st.session_state.a4_pdfs = None
                except Exception as e:
                    st.error(f"前処理でエラーが発生しました: {e}")
    else:
        target_col, target, date_col = "(なし)", None, None

with col_view:
    st.header("📊 データビュー")
    if st.session_state.df is None:
        st.info("👈 左側のメニューからファイルを読み込んでください。")
    else:
        is_processed = st.session_state.processed_df is not None
        target_df = st.session_state.processed_df if is_processed else st.session_state.df
        num_cols_list = get_numeric_columns(target_df)
        view_tabs = st.tabs(["データ確認 ＆ ダウンロード", "プロファイル", "📊 グループ集計", "📈 時系列グラフ", "🔥 Yとの相関", "🤖 機械学習", "📄 A4サマリー"])

        with view_tabs[0]:
            src = source_of(target_df)
            n_test = int((src == "test").sum())
            if is_processed:
                st.success("✅ 前処理が完了しました！")
                if st.session_state.orig_test_rows and n_test != st.session_state.orig_test_rows:
                    st.error(f"⚠ testの行数が {st.session_state.orig_test_rows} → {n_test} に変わっています。提出でエラーになります。")
                with st.expander("📝 処理ログ（どの処理で行・列が変わったか）", expanded=False):
                    st.dataframe(pd.DataFrame(st.session_state.logs), use_container_width=True)
            display_df = target_df.drop(columns=["_source_"], errors="ignore")
            st.markdown(f"**データサイズ: {display_df.shape[0]}行（train {int((src != 'test').sum())} / test {n_test}） × {display_df.shape[1]}列**")
            st.dataframe(display_df.head(1000), height=400, use_container_width=True)
            if is_processed and st.button("現在の結果をベースに設定を続ける", use_container_width=True):
                st.session_state.df = target_df; st.session_state.processed_df = None; st.session_state.logs = []; st.rerun()

            st.divider()
            save_format = st.radio("**保存形式を選択**", ["通常のCSV (.csv)", "GZIP圧縮CSV (.csv.gz)", "Parquet形式 (.parquet)"], horizontal=True,
                                   help="v38ノートに渡すなら Parquet がおすすめ（category型・日付型がそのまま残る）")
            export_header = not st.checkbox("CSVのヘッダーを含めずにダウンロード", value=False)
            if "train" in src.values and "test" in src.values:
                tr_b, ext, mime = convert_df(target_df[src == "train"].drop(columns=["_source_"]), save_format, export_header)
                te_b, _, _ = convert_df(target_df[src == "test"].drop(columns=["_source_"]), save_format, export_header)
                d1, d2, d3 = st.columns(3)
                with d1: st.download_button("TrainデータをDL", tr_b, f"preprocessed_train.{ext}", mime, use_container_width=True)
                with d2: st.download_button("TestデータをDL", te_b, f"preprocessed_test.{ext}", mime, use_container_width=True)
                with d3:
                    zb = io.BytesIO()
                    with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
                        z.writestr(f"preprocessed_train.{ext}", tr_b); z.writestr(f"preprocessed_test.{ext}", te_b)
                        if st.session_state.sample_raw is not None:
                            nm = st.session_state.sample_name
                            z.writestr(nm if "sample" in nm.lower() else f"sample_{nm}", st.session_state.sample_raw)
                    st.download_button("ZIPでまとめてDL", zb.getvalue(), "preprocessed_data.zip", "application/zip", use_container_width=True, type="primary")
                st.caption("ZIPは v38ノートの ZIP_PATH にそのまま指定できます" + ("（sampleも同梱）" if st.session_state.sample_raw is not None else ""))
            else:
                out_b, ext, mime = convert_df(target_df.drop(columns=["_source_"], errors="ignore"), save_format, export_header)
                st.download_button("**ダウンロード**", out_b, f"preprocessed_data.{ext}", mime, type="primary")

            st.divider()
            st.markdown("##### 📄 A4サマリーPDF（データの状況確認）")
            st.caption("①基本情報表 ②グラフ一覧 ③相関 ④目的変数×時系列 ⑤目的変数との関係。目的変数・日付列は左の設定を使います（testは目的変数が空なので①〜③のみ）")
            a4c1, a4c2 = st.columns([6, 4])
            with a4c1: a4_all = st.checkbox("train+test合体のPDFも作る", value=False)
            with a4c2: a4_go = st.button("PDFを作成", use_container_width=True)
            if a4_go:
                with st.spinner("PDF作成中..."):
                    try:
                        src_now = source_of(target_df)
                        fr = {k: target_df[src_now == k].drop(columns=["_source_"]).reset_index(drop=True)
                              for k in ("train", "test", "single") if (src_now == k).any()}
                        if "single" in fr: fr = {"data": fr["single"]}
                        st.session_state.a4_pdfs = make_a4_pdf(fr, target, date_col, a4_all)
                    except Exception as e:
                        st.error(f"PDF作成でエラー: {e}")
            for fname, b in (st.session_state.get("a4_pdfs") or {}).items():
                st.download_button(f"⬇ {fname}", b, fname, "application/pdf", use_container_width=True, key=f"dl_{fname}")

        with view_tabs[1]:
            st.info("💡 「データ型」が `category` と表示されていれば変換成功です。")
            st.dataframe(profile_table(target_df), use_container_width=True)

        with view_tabs[2]:
            cols_clean = [c for c in target_df.columns if c != "_source_"]
            e1, e2 = st.columns(2)
            with e1: grp_col = st.selectbox("グループ化する列 (曜日など)", ["(なし)"] + cols_clean)
            with e2: tgt_col = st.selectbox("集計する数値列 (販売数など)", ["(なし)"] + num_cols_list)
            if grp_col != "(なし)" and tgt_col != "(なし)":
                try: st.dataframe(pd.to_numeric(target_df[tgt_col], errors="coerce").groupby(target_df[grp_col], observed=True)
                                  .agg(["count", "sum", "mean", "median", "std"]).round(1), use_container_width=True)
                except Exception as e: st.warning(f"集計できませんでした: {e}")

        with view_tabs[3]:
            t1, t2 = st.columns([3, 7])
            cols_clean = [c for c in target_df.columns if c != "_source_"]
            with t1: date_col_ts = st.selectbox("X軸 (日付/時間の列)", ["(なし)"] + cols_clean,
                                                index=cols_clean.index(date_col) + 1 if date_col in cols_clean else 0)
            with t2: val_cols_ts = st.multiselect("Y軸 (数値列を複数選択)", num_cols_list)
            if date_col_ts != "(なし)" and val_cols_ts:
                try:
                    tmp = target_df.copy(); tmp[date_col_ts] = pd.to_datetime(tmp[date_col_ts], errors='coerce')
                    tmp = tmp.dropna(subset=[date_col_ts]).sort_values(by=date_col_ts)
                    if tmp.empty: st.warning("有効な日付データが見つかりませんでした。")
                    else:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        for c in val_cols_ts: ax.plot(tmp[date_col_ts], pd.to_numeric(tmp[c], errors='coerce'), label=c, marker='o', markersize=3, alpha=0.8)
                        ax.set_xlabel(date_col_ts); ax.legend(); ax.grid(True, alpha=0.3); fig.autofmt_xdate(); fig.tight_layout()
                        st.pyplot(fig)
                except Exception as e: st.error(f"グラフの描画に失敗しました: {e}")

        with view_tabs[4]:
            if target_col == "(なし)":
                st.warning("👈 左側のメニューで「目的変数」を選択してください。")
            else:
                trn = target_df[train_mask(target_df)]
                y_series = pd.to_numeric(trn[target_col], errors='coerce')
                feat_cols = [c for c in num_cols_list if c != target_col]
                if feat_cols:
                    corr_cols = st.multiselect("相関を確認する特徴量", feat_cols, default=feat_cols[:min(30, len(feat_cols))])
                    if corr_cols:
                        corrs = trn[corr_cols].apply(pd.to_numeric, errors='coerce').corrwith(y_series)
                        cdf = pd.DataFrame({"特徴量": corrs.index, "相関係数": corrs.values}).dropna()
                        if not cdf.empty:
                            cdf = cdf.reindex(cdf['相関係数'].abs().sort_values(ascending=False).index).reset_index(drop=True)
                            cdf['相関係数'] = cdf['相関係数'].round(3)
                            try: st.dataframe(cdf.style.background_gradient(cmap='coolwarm', subset=['相関係数'], vmin=-1, vmax=1), use_container_width=True)
                            except Exception: st.dataframe(cdf, use_container_width=True)
                        else: st.warning("有効な相関が計算できませんでした。")

        # ------------------------- 🤖 機械学習 -------------------------
        with view_tabs[5]:
            st.markdown("### 🤖 機械学習（v38ノートと同じ方式でその場で検証）")
            st.caption("今の表（前処理後があればそちら）で学習。時系列CV → 上位K個の平均 → 末尾期間の重要度 → 全データで再学習してtestを予測")
            if not target:
                st.warning("👈 左側で「🎯 目的変数」を選んでください。")
            elif pd.to_numeric(target_df.loc[train_mask(target_df), target], errors="coerce").notna().sum() < 20:
                st.warning("目的変数に数値が20行以上ありません（このタブは数値を予測する回帰のみ対応）。")
            else:
                if not HAS_LGBM: st.caption("※ lightgbm が入っていないので LightGBM 系は表示されません（requirements.txt に lightgbm を追加）")
                all_models = list(model_candidates("rmse").keys())
                v38_default = [m for m in ['GBM | 浅い木(深さ2)', 'SVR'] if m in all_models]
                feat_choices = [c for c in target_df.columns if c not in ("_source_", target)]
                with st.form("ml_form"):
                    a1, a2 = st.columns(2)
                    with a1:
                        metric = st.radio("採点の指標", ["rmse", "mae"], horizontal=True, help="提出の採点に合わせる。RMSE=大外れを重く見る")
                        models = st.multiselect("比べるモデル（空なら全部）", all_models, default=v38_default)
                        top_k = st.number_input("上位何個を平均するか（0=CVで自動）", 0, 12, 2)
                    with a2:
                        exclude = st.multiselect("学習に使わない列", feat_choices)
                        include_date = st.checkbox("日付列を特徴量にも入れる（v38既定ON）", value=True, disabled=not date_col)
                        min_flag = st.number_input("1がこの行数未満の0/1列は使わない（0=しない）", 0, 1000, 5)
                        data_name = st.text_input("データ名（実験捜査ファイルでの分類）", value=st.session_state.data_name)
                    with st.expander("詳しい設定（水準の補正・CV）"):
                        b1, b2 = st.columns(2)
                        with b1:
                            level_mode = st.selectbox("水準の補正（売上が時期で上下するデータ向け）", ["なし", "ratio", "diff"],
                                                      help="お弁当データは ratio。ratio=割って戻す / diff=引いて戻す")
                            level_last = st.number_input("直近の水準＝最後の何行の平均", 1, 500, 20)
                            level_win = st.number_input("水準を測る移動平均の行数", 3, 500, 21)
                        with b2:
                            folds = st.number_input("時系列CVの回数", 2, 10, 3)
                            horizon = st.number_input("1回で予測する行数（0=testの行数）", 0, 10000, 0)
                            clip0 = st.checkbox("予測を0以上にする", value=True)
                            perm = st.checkbox("重要度（Permutation）を計算", value=True)
                    go = st.form_submit_button("▶ 学習して検証する", type="primary", use_container_width=True)

                C = dict(target=target, date_col=date_col, metric=metric, models=models, top_k=int(top_k), exclude=exclude,
                         include_date=include_date and bool(date_col), min_flag=int(min_flag),
                         level={"mode": None if level_mode == "なし" else level_mode, "window": int(level_win), "last_n": int(level_last)},
                         folds=int(folds), horizon=int(horizon), clip0=clip0, perm=perm)
                if go:
                    with st.spinner("学習中...（数十秒かかることがあります）"):
                        try:
                            res = run_ml(target_df, C)
                            st.session_state.ml_result = (res, C)
                            st.session_state.data_name = data_name or "未分類"
                            now = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9)))
                            hold = metric_values(res["hold_df"]["実績"], res["hold_df"]["予測"])
                            rec = {
                                "id": "app-" + now.strftime("%Y%m%d-%H%M%S"), "time": now.strftime("%Y-%m-%dT%H:%M:%S"),
                                "project": data_name or "未分類", "source": "app v4", "metric": metric.upper(),
                                "cv": round(res["cv_score"], 6), "submit": None,
                                "models": " + ".join(res["final_names"]), "k": res["k"],
                                "features": [str(c) for c in res["feat"]], "excluded": [str(c) for c in exclude],
                                "n_features": len(res["feat"]),
                                "holdout": {"検証RMSE": round(hold["RMSE"], 6), "検証MAE": round(hold["MAE"], 6)},
                                "importance": ({str(r_["特徴量"]): round(float(r_["重要度平均"]), 6) for _, r_ in res["imp"].head(20).iterrows()}
                                               if res["imp"] is not None else None),
                                "settings": {"LEVEL_NORMALIZE": C["level"]["mode"], "LEVEL_LAST_N": C["level"]["last_n"],
                                             "LEVEL_WINDOW": C["level"]["window"], "INCLUDE_DATE_AS_FEATURE": C["include_date"],
                                             "MODELS_TO_USE": list(models), "ENSEMBLE_TOP_K": C["top_k"] or None, "CV_FOLDS": C["folds"],
                                             "CV_HORIZON": C["horizon"] or None, "MIN_FLAG_COUNT": C["min_flag"],
                                             "DATE_COL": date_col, "TARGET": target},
                                "memo": ""}
                            st.session_state.ml_records.append(rec)
                            st.session_state.ml_history.append({
                                "時刻": _dt.datetime.now().strftime("%m/%d %H:%M"), f"CV_{res['key']}": round(res["cv_score"], 4),
                                "モデル": " + ".join(res["final_names"]), "K": res["k"], "特徴量数": len(res["feat"]),
                                "除外": ", ".join(map(str, exclude)), "水準補正": level_mode, "日付を特徴量": C["include_date"],
                                "提出スコア(手入力)": float("nan")})
                        except Exception as e:
                            st.error(f"学習でエラー: {e}")

                if st.session_state.ml_result:
                    res, Cr = st.session_state.ml_result
                    key = res["key"]
                    st.success(f"CV {key}（上位{res['k']}個の平均）: **{res['cv_score']:.4f}**　/　使ったモデル: {' + '.join(res['final_names'])}")
                    st.caption(f"学習 {res['train_rows']}行 / 特徴量 {len(res['feat'])}個 / CV {res['n_folds']}回 × {res['horizon']}行")
                    for m_ in res["msgs"]: st.caption("・" + m_)
                    st.markdown("**モデルごとのCV成績**"); st.dataframe(res["summary"].round(4), use_container_width=True)
                    with st.expander("上位K個の平均の成績"): st.dataframe(res["ens"].round(4), use_container_width=True)

                    st.markdown("**末尾期間の実績 vs 予測**")
                    h = res["hold_df"]
                    fig, ax = plt.subplots(figsize=(10, 3.5))
                    ax.plot(h['日付'], h['実績'], marker='o', ms=3, label='実績'); ax.plot(h['日付'], h['予測'], marker='o', ms=3, label='予測')
                    ax.legend(); ax.grid(alpha=0.3); fig.autofmt_xdate(); fig.tight_layout(); st.pyplot(fig)
                    h2 = h.assign(誤差=h['予測'] - h['実績'])
                    with st.expander("誤差の大きい行"):
                        st.dataframe(h2.reindex(h2['誤差'].abs().sort_values(ascending=False).index).head(15).round(2), use_container_width=True)

                    if res["imp"] is not None:
                        st.markdown("**特徴量の重要度（末尾期間・Permutation）**　※マイナスでも、消して良くなるとは限らない（1つ消して提出で確認）")
                        st.dataframe(res["imp"].round(5), use_container_width=True, height=300)

                    if res["test_pred"] is not None:
                        sample_df = None
                        if st.session_state.sample_raw is not None:
                            has_h = st.checkbox("提出見本の1行目は見出し（ヘッダー）", value=False)
                            sample_df = read_sample_bytes(st.session_state.sample_raw, st.session_state.sample_name, has_h)
                        try:
                            sub_bytes, hdr = build_submission(res, sample_df, target, date_col)
                            st.download_button("📤 提出用CSV（submission.csv）をDL", sub_bytes, "submission.csv", "text/csv", type="primary", use_container_width=True)
                            st.caption("見本の形に合わせました" if sample_df is not None else "見本がないので「日付, 予測値」（ヘッダーなし）で出力")
                        except Exception as e:
                            st.error(str(e))

                    st.markdown("**📋 v38ノートの設定セルに貼る設定**（同じ条件でColabでも回せます）")
                    st.code("\n".join([
                        f"EXCLUDE_COLS = {list(Cr['exclude'])!r}",
                        f"LEVEL_LAST_N = {Cr['level']['last_n']}",
                        f"MODELS_TO_USE = {list(Cr['models'])!r}",
                        f"ENSEMBLE_TOP_K = {Cr['top_k'] or None}",
                        f"INCLUDE_DATE_AS_FEATURE = {Cr['include_date']}",
                        f"TARGET = {target!r}",
                        f"DATE_COL = {date_col!r}",
                        f"EVAL_METRIC = {Cr['metric']!r}",
                        f"LEVEL_NORMALIZE = {Cr['level']['mode']!r}",
                        f"LEVEL_WINDOW = {Cr['level']['window']}",
                        f"MIN_FLAG_COUNT = {Cr['min_flag']}",
                        f"CV_FOLDS = {Cr['folds']}",
                        f"CV_HORIZON = {Cr['horizon'] or None}",
                    ]), language="python")
                    if st.session_state.ml_records:
                        st.markdown("**🗂 実験記録**（右上のコピーボタンで写して、実験捜査ファイルの「記録を追加」に貼る）")
                        st.code(json.dumps(st.session_state.ml_records[-1], ensure_ascii=False), language="json")

                if st.session_state.ml_history:
                    st.markdown("**🧪 実験履歴**（1回に1つだけ変えて比べる。提出スコアは表に手で書き込めます）")
                    hist = st.data_editor(pd.DataFrame(st.session_state.ml_history), use_container_width=True, key="hist_editor")
                    hc1, hc2 = st.columns(2)
                    with hc1:
                        st.download_button("履歴をCSVでDL", hist.to_csv(index=False).encode("utf-8-sig"), "ml_history.csv", "text/csv", use_container_width=True)
                        if st.session_state.ml_records:
                            st.download_button("実験記録をまとめてDL（.jsonl）",
                                               "\n".join(json.dumps(r_, ensure_ascii=False) for r_ in st.session_state.ml_records).encode("utf-8"),
                                               "experiment_log_app.jsonl", "application/json", use_container_width=True)
                    with hc2:
                        if st.button("履歴を消す", use_container_width=True):
                            st.session_state.ml_history = []; st.session_state.ml_records = []; st.session_state.pop("hist_editor", None); st.rerun()

        # ------------------------- 📄 A4サマリー（画面表示） -------------------------
        with view_tabs[6]:
            st.markdown("### 📄 A4サマリー（画面で見る）")
            st.caption("A4サマリーノート v7 と同じ図を、PDFにせずここに表示します。PDFが欲しいときは「データ確認」タブの「PDFを作成」")
            if _jp_font is None:
                st.warning("日本語フォントが見つからないため、図の日本語が「□」になります。packages.txt（fonts-noto-cjk）を一緒に置いてください。")
            src_now = source_of(target_df)
            parts = [k for k in ("train", "test", "single") if (src_now == k).any()]
            s1, s2 = st.columns(2)
            with s1: which = st.radio("どのデータ", parts, horizontal=True, key="a4_which")
            with s2: pages_sel = st.multiselect("表示するページ", ["④目的変数×時系列", "⑤目的変数との関係", "②グラフ一覧", "③相関"],
                                                default=["④目的変数×時系列"], key="a4_pages")
            if st.button("図を表示", use_container_width=True, key="a4_show"):
                dfv = target_df[src_now == which].drop(columns=["_source_"]).reset_index(drop=True)
                globals().update(TIME_COL=date_col or "", TARGET=target or "-")
                tr = None
                if which != "test" and (src_now == "test").any() and date_col:
                    tt = pd.to_datetime(target_df.loc[src_now == "test", date_col], errors="coerce").dropna()
                    if len(tt): tr = (tt.min(), tt.max())

                class _Grab:  # PdfPages の代わりに、作った図を受け取って画面に出す
                    def savefig(self, fig): st.pyplot(fig)
                g = _Grab()
                with st.spinner("作図中..."):
                    try:
                        if "④目的変数×時系列" in pages_sel:
                            if not find_target(dfv): st.info("このデータには目的変数の値がありません（testは空なので④⑤は出ません）")
                            elif draw_target_overview(dfv, which, g, None, tr): draw_target_feature_pages(dfv, which, g, None, tr)
                            else: st.info("日付列が見つからないため④は出せません。左の「📅日付列」を選んでください")
                        if "⑤目的変数との関係" in pages_sel and find_target(dfv): draw_target_relation_pages(dfv, which, g)
                        if "②グラフ一覧" in pages_sel: draw_graph_pages(dfv, which, g)
                        if "③相関" in pages_sel: draw_corr_page(dfv, which, g)
                    except Exception as e:
                        st.error(f"作図でエラー: {e}")
                    plt.close("all")
