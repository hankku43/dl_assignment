import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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
# D.1. 分離特徵(X)和目標(y)
X = df_final.drop("Survived", axis=1)
y = df_final["Survived"]

# D.2. 創建訓練集、測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("已創建訓練集、測試集")

# -------------------------------------------
#              E.資料標準化
# -------------------------------------------
print("\n開始進行資料標準化")
scaler = StandardScaler()

# E.1. 定義需要標準化的數值欄位
numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
print(f"將對以下欄位進行標準化：{numeric_cols}")

# E.2. 訓練集標準化 (Fit and Transform)
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
print("訓練集數值欄位標準化完成。")

# E.3. 測試集標準化 (Transform ONLY)
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
print("測試集數值欄位標準化完成。")


# -------------------------------------------
#              F.建立Dataset
# -------------------------------------------
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # 轉成 (N, 1) 形狀

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_dataset = TitanicDataset(X_train, y_train)
test_dataset = TitanicDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -------------------------------------------
#              G.定義模型
# -------------------------------------------
class SurvivalModel(nn.Module):
    def __init__(self):
        super(SurvivalModel, self).__init__()

        # 輸入層有 2 個特徵 (Gender, Height)
        self.h_layer1 = nn.Linear(2, 2)
        # 輸出層只有 1 個值 (Weight)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = self.h_layer(x)
        x = self.output(x)  # 直接輸出預測值
        return x
#123456