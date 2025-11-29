import re
import pandas as pd

df = pd.read_csv("titanic.csv")
print("=====原始數據讀取完成====")
# ===========================================
#              數據前處理
# ===========================================

# -------------------------------------------
#              A.檢查缺失
# -------------------------------------------
print("\n===================")
print("====開始檢查資料缺失====")
# A1. 計算數量和比例
missing_count = df.isnull().sum()
missing_ratio = df.isnull().mean() * 100

# A2. 合併並格式化輸出
missing_summary = pd.DataFrame(
    {"缺失數量": missing_count, "缺失比例 (%)": missing_ratio}
)

# A3. 只顯示有缺失的欄位
missing_summary = missing_summary[missing_summary["缺失數量"] > 0]

print("=" * 30)
print(missing_summary)
print("=" * 30)
print("Age 以中位數填補")
print("Cabin 標註缺失")
print("Embarked 以眾數填補")
print()

# -------------------------------------------
#              B.處理缺失
# -------------------------------------------
print("\n===================")
print("====開始處理缺失====")

# B1. 複製資料
df_cleaned = df.copy()

# B2. Age 缺失處理
median_age = df_cleaned["Age"].median()
print(f"Age 欄位中位數: {median_age:.1f}")
df_cleaned["Age"] = df_cleaned["Age"].fillna(median_age)

# B3. Cabin 缺失處理
df_cleaned["Cabin"] = df_cleaned["Cabin"].fillna("Missing")
print(f"Cabin 欄位缺失以 'Missing' 填補")

# B4. Embarked 缺失處理
mode_embarked = df_cleaned["Embarked"].mode()[0]
print(f"Embarked 欄位眾數: {mode_embarked}")
df_cleaned["Embarked"] = df_cleaned["Embarked"].fillna(mode_embarked)

print("缺失填補完畢")

# -------------------------------------------
#              C.Encoding
# -------------------------------------------

print("\n===================")
print("====開始Encoding====")
# C1. 資料複製
df_processed = df_cleaned.copy()

# C2. 新增稱謂(從Name提取)

print("從Name提取稱謂(Title)")
df_processed["Title"] = df_processed["Name"].apply(
    # 使用正規表達式提取 . 之前的文字
    lambda x: re.search(r"([A-Za-z]+)\.", x).group(1)
)

# C3. 處理少見稱謂
print("【Title 欄位初始統計結果】")
# C3.1. 計算絕對數量
title_counts = df_processed["Title"].value_counts()

# C3.2. 計算比例 (設定 normalize=True)
title_percentages = df_processed["Title"].value_counts(normalize=True)

# C3.3. 結合成一個 DataFrame 進行展示
title_summary = pd.DataFrame(
    {
        "數量 (Count)": title_counts,
        "比例 (%)": (title_percentages * 100).round(2),  # 將比例四捨五入到小數點後兩位
    }
)
print(title_summary)

# C3.4. 篩選出比例< 0.5 % 的稱謂
rare_titles_list = title_percentages[title_percentages < 0.005].index.tolist()

# C3.5. 置換成 Rare
df_processed["Title"] = df_processed["Title"].replace(rare_titles_list, "Rare")
print("完成少見稱位(<0.5%)置換為'Rare'")

# C4. 提取 Cabin 欄位的第一個字母作為甲板代號
print("提取 Cabin 欄位的第一個字母作為甲板代號")
df_processed["Cabin"] = df_processed["Cabin"].str[0]

# C5. 移除不需使用的欄位
print("移除不需使用的欄位(['PassengerId', 'Name', 'Ticket'])")
cols_to_drop = ["PassengerId", "Name", "Ticket"]
df_processed = df_processed.drop(cols_to_drop, axis=1)


# C6. 獨熱編碼
print("\n開始進行獨熱編碼(One-Hot Encoding)")
print("參與欄位(['Title', 'Sex', 'Embarked', 'Pclass', 'Cabin'])")
categorical_cols = ["Title", "Sex", "Embarked", "Pclass", "Cabin"]

# C6.1. 執行獨熱編碼 (drop_first=True 避免共線性)
df_final = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

print("類別編碼完成。")
print(f"最終數據集欄位：{df_final.columns.tolist()} (已大幅擴充)")


# -------------------------------------------
#              D.測試集拆分
# -------------------------------------------
print("開始進行測試集拆分")
