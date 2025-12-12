import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
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

# D.3. 強制重置索引
# 防止 StandardScaler 賦值時的 IndexError
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# -------------------------------------------
#       E.資料標準化 (修正版)
# -------------------------------------------
print("\n開始進行資料標準化")
scaler = StandardScaler()

# E.1. 定義需要標準化的數值欄位
numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
print(f"將對以下欄位進行標準化：{numeric_cols}")

# E.2. 訓練集標準化 (Fit and Transform)
# E.2.1. 對數據進行轉換 (輸出為 NumPy Array)
X_train_scaled = scaler.fit_transform(X_train[numeric_cols])

# E.2.2 將 NumPy Array 轉換回帶有原始欄位名稱的 DataFrame，並賦值回 X_train
X_train[numeric_cols] = pd.DataFrame(X_train_scaled, columns=numeric_cols, index=X_train.index)

print("訓練集數值欄位標準化完成。")

# E.3. 測試集標準化 (Transform ONLY)
# E.3.1: 對數據進行轉換 (輸出為 NumPy Array)
X_test_scaled = scaler.transform(X_test[numeric_cols])

# E.3.2 將 NumPy Array 轉換回帶有原始欄位名稱和索引的 DataFrame，並賦值回 X_test
X_test[numeric_cols] = pd.DataFrame(X_test_scaled, columns=numeric_cols, index=X_test.index)

print("測試集數值欄位標準化完成。")


# -------------------------------------------
#              F.建立Dataset
# -------------------------------------------
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        try:
            X_numpy = X.astype(float).values
            y_numpy = y.astype(float).values
        except ValueError as e:
            print("資料轉換失敗！請檢查 X_train 中是否還有字串欄位。")
            print("目前的欄位型態：")
            print(X.dtypes)
            raise e

        self.X = torch.tensor(X_numpy, dtype=torch.float32)
        self.y = torch.tensor(y_numpy, dtype=torch.float32).view(-1, 1)
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
print("定義模型")
class SurvivalModel(nn.Module):
    def __init__(self, input_num):
        super(SurvivalModel, self).__init__()
        
        # 增加寬度以避免瓶頸
        self.layer1 = nn.Linear(input_num, 8) 
        self.layer2 = nn.Linear(8, 4)
        self.layer3 = nn.Linear(4, 1) 
        
        self.relu = nn.ReLU()

    def forward(self, x):
        
        # Layer 1: Linear -> ReLU
        x = self.layer1(x)
        x = self.relu(x)
        
        # Layer 2: Linear -> ReLU
        x = self.layer2(x)
        x = self.relu(x)
        
        # Output Layer: Linear
        x = self.layer3(x)

        return x
    
# -------------------------------------------
#              H.開始訓練
# -------------------------------------------
print("開始訓練")

# H1. 實例model
input_features = X_train.shape[1] 
model = SurvivalModel(input_features)

# H2. 設定訓練模式
model.train()

# H3. 定義損失函數
criterion = nn.BCEWithLogitsLoss() 
print("損失函數已定義為 nn.BCEWithLogitsLoss()")

# H4. 設定學習率
learning_rate = 0.001

# H5. 優化器：使用 Adam，並將模型的參數傳入
optimizer = optim.SGD(model.parameters(), lr=learning_rate) 
print(f"優化器已定義為 Adam, 學習率: {learning_rate}")


# H6. 開始訓練
num_epochs = 1000
for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        
        # 步驟 1: 梯度歸零
        optimizer.zero_grad()
        
        # 步驟 2: 前向傳播
        predictions = model(batch_X)
        
        # 步驟 3: 計算損失
        loss = criterion(predictions, batch_y)
        
        # 步驟 4: 反向傳播
        loss.backward()
        
        # 步驟 5: 更新參數
        optimizer.step()
        
        running_loss += loss.item()

    # 打印每個 epoch 的平均 Loss
    avg_loss = running_loss / len(train_loader)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}"
    )


# -------------------------------------------
#       I.開始測試 (修正版)
# -------------------------------------------
print("開始測試")

# I1. 切換到評估模式
model.eval()

# 初始化總計數器
total_correct = 0
total_samples = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        
        # 1. 執行前向傳播 (輸出 Logits)
        predicted_logits = model(batch_X)

        # 2. 將 Logits 轉換為機率 (Sigmoid)
        predicted_probs = torch.sigmoid(predicted_logits)
        
        # 3. 將機率轉換為二元標籤 (> 0.5 為 1，否則為 0)
        # .float() 確保類型一致
        predicted_labels = (predicted_probs > 0.5).float()
        
        # 4. 計算批次正確數
        total_correct += (predicted_labels == batch_y).sum().item()
        total_samples += batch_y.size(0)

# 6. 計算最終準確度
accuracy = total_correct / total_samples
print(f"測試結果: 準確度 {accuracy:.4f} ({total_correct}/{total_samples})")