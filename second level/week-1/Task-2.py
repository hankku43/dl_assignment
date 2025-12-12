import glob
import os
import re
import pandas as pd

FILE_PATTERN = "ptt_*.csv"
CLEANED_PREFIX = "cleaned_"
TITLE_COL = "Article Title"

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    資料清理邏輯：
    1. 確保 Article Title 欄位存在
    2. 去除所有欄位的前後空白
    3. 刪除包含 [公告] 標籤的行
    4. 刪除包含 Re: 或 Fw: 的行（任何位置）
    5. 移除所有 [] 及其內容（副分類標籤）
    6. 將英文大寫轉為小寫
    7. 清理前後空白
    8. 刪除空標題的行
    """
    
    # 確保 title 欄位存在
    if TITLE_COL not in df.columns:
        return df

    # 步驟 2: 去除所有欄位的前後空白
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].apply(lambda col: col.astype(str).str.strip())

    # 步驟 3: 刪除 [公告] 標籤的行
    mask_announcement = df[TITLE_COL].fillna("").str.contains(r'\[公告\]', case=False, regex=True)
    df = df[~mask_announcement]

    # 步驟 4: 刪除包含 Re: 或 Fw: 的行
    pattern = re.compile(r'(?:re|r|fw)\s*[:：]', re.I)
    mask_remove = df[TITLE_COL].fillna("").apply(lambda t: bool(pattern.search(t)))
    df = df[~mask_remove]

    # 步驟 5: 移除所有 [] 及其內容
    df[TITLE_COL] = df[TITLE_COL].str.replace(r'\[.*?\]', '', regex=True)
    
    # 步驟 6: 英文大寫轉小寫
    df[TITLE_COL] = df[TITLE_COL].str.lower()
    
    # 步驟 7: 清理前後空白
    df[TITLE_COL] = df[TITLE_COL].str.strip()
    
    # 步驟 8: 刪除標題包含「已刪」且內文長度不足 10 字的行（若有內容欄位）
    mask_title_deleted = df[TITLE_COL].fillna("").str.contains("已刪", na=False)
    mask_short_title = df[TITLE_COL].fillna("").str.len() < 10
    df = df[~(mask_title_deleted & mask_short_title)]

    # 步驟 9: 刪除空標題的行
    df = df[df[TITLE_COL].astype(bool)]

    return df

def process_files():
    """
    主處理函式：
    - 找到所有符合 FILE_PATTERN 的檔案
    - 逐一清理並輸出帶有 CLEANED_PREFIX 前綴的新檔
    """
    files = sorted(glob.glob(FILE_PATTERN))
    if not files:
        print("No files found matching pattern:", FILE_PATTERN)
        return

    for path in files:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        before = len(df)
        df_clean = clean_dataframe(df)
        after = len(df_clean)

        base = os.path.basename(path)
        out_path = os.path.join(os.path.dirname(path), CLEANED_PREFIX + base)
        try:
            df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"Processed {base}: {before} -> {after}. Saved to {os.path.basename(out_path)}")
        except Exception as e:
            print(f"Failed to write {out_path}: {e}")

if __name__ == "__main__":
    process_files()