import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 步驟 1: 讀取與預處理資料 ---

# 模擬讀取您的 CSV
# 如果檔案名是其他名稱，請修改這裡
df = pd.read_csv("gender-height-weight.csv")

# 1.1 處理類別資料 (Encoding)
# 將 Male 設為 0, Female 設為 1 (或是用 LabelEncoder)
# 注意：CSV 讀進來後，Pandas 通常會自動處理掉雙引號
mapping = {"Male": 0, "Female": 1}
df["Gender"] = df["Gender"].map(mapping)

# 檢查是否有無法轉換的值 (NaN)
if df["Gender"].isnull().any().any():
    print("警告：發現未知的性別標籤，請檢查 CSV 內容")
    df.dropna(inplace=True)  # 移除資料

# 1.2 分離特徵 (X) 與 目標 (y)
X_raw = df[["Gender", "Height"]].values  # 輸入：性別、身高 (2個特徵)
y_raw = df["Weight"].values  # 輸出：體重

# 1.3 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# 1.4 標準化 (Normalization)
# 雖然 Gender 是 0/1，但 Height 是 60-80，Weight 是 100-200
# 數值差異大會導致訓練困難，建議對「身高」進行標準化
# 為了方便，這裡我們對 X 整體做標準化 (0/1 的 Gender 被標準化也沒關係)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

# --- 步驟 2: 建立 Dataset ---


class GenderHeightWeightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # 轉成 (N, 1) 形狀

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = GenderHeightWeightDataset(X_train, y_train)
test_dataset = GenderHeightWeightDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --- 步驟 3: 定義模型 (Regression Model) ---


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()

        # 輸入層有 2 個特徵 (Gender, Height)
        self.h_layer = nn.Linear(2, 2)
        # 輸出層只有 1 個值 (Weight)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = self.h_layer(x)
        x = self.output(x)  # 直接輸出預測值
        return x


model = RegressionModel()

# --- 複製所有層的初始權重和偏置 ---
# 1. 隱藏層 (h_layer)
initial_h_weight = model.h_layer.weight.clone().detach().cpu().numpy()
initial_h_bias = model.h_layer.bias.clone().detach().cpu().numpy()

# 2. 輸出層 (output)
initial_out_weight = model.output.weight.clone().detach().cpu().numpy()
initial_out_bias = model.output.bias.clone().detach().cpu().numpy()

# --- 步驟 4: 定義 Loss 與優化器 ---

# 回歸任務最常用 MSELowss (Mean Squared Error)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# --- 步驟 4.5: 檢查初始損失 ---

model.eval()  # 切換到評估模式
initial_loss_sum = 0.0

print("\n檢查初始損失...")

with torch.no_grad():
    for batch_X, batch_y in train_loader:
        # 前向傳播
        predictions = model(batch_X)

        # 計算損失 (使用 MSELoss，單位是標準差平方)
        loss = criterion(predictions, batch_y)

        # 累積損失總和
        initial_loss_sum += loss.item()

# 計算平均初始損失
avg_initial_loss = initial_loss_sum / len(train_loader)
print(f"✅ 訓練前初始平均損失 (MSE): {avg_initial_loss:.4f}")
print("----------------------------------------")

# --- 步驟 5: 開始訓練 ---

num_epochs = 20

print("開始訓練...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
    )


# 最終權重
final_h_weight = model.h_layer.weight.clone().detach().cpu().numpy()
final_h_bias = model.h_layer.bias.clone().detach().cpu().numpy()
final_out_weight = model.output.weight.clone().detach().cpu().numpy()
final_out_bias = model.output.bias.clone().detach().cpu().numpy()


print("\n--- 隱藏層 h_layer (2 -> 4) ---")
# 使用訓練前複製的變數
print(f"初始權重 (W_h): \n{initial_h_weight}")
print(f"最終權重 (W_h): \n{final_h_weight}")
print(f"最終偏置 (b_h): \n{final_h_bias}")


print("\n--- 輸出層 output (4 -> 1) ---")
print("權重 (W_out) - [1, 4] 向量:")
# 使用訓練前複製的變數
print(f"初始: {initial_out_weight}")
# 使用訓練後取得的變數
print(f"最終: {final_out_weight}")

print("\n偏置 (b_out) - [1] 數值:")
# 使用訓練前複製的變數
print(f"初始: {initial_out_bias}")
# 使用訓練後取得的變數
print(f"最終: {final_out_bias}")

# --- 步驟 6: 簡單測試與預測 ---
print("開始測試...")
model.eval()  # 切換到評估模式

total_abs_error = 0.0
total_samples = 0.0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        predicted = model(batch_X)

        abs_error = torch.abs(predicted - batch_y)

        total_abs_error += abs_error.sum().item()

        total_samples += batch_X.size(0)

    avg_mae_normalized = total_abs_error / total_samples
    mae_in_pounds = avg_mae_normalized * scaler.scale_[0]
    print("測試結果...")
    print(f"實際平均誤差 (MAE): {mae_in_pounds:.2f} 磅")
