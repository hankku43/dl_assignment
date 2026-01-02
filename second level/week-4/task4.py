import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from gensim.models import Doc2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# 設定中文字型 (選用，若你的看板名稱包含中文需要打開這行，否則 matplotlib 會顯示亂碼)
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
# plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 設定參數
# ==========================================
MODEL_PATH = "ptt_titles_doc2vec_ver3.model"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
TEST_SIZE = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 直接從 Doc2Vec 提取資料 (維持不變)
# ==========================================
print("Loading Doc2Vec model...")
d2v_model = Doc2Vec.load(MODEL_PATH)

X_data = []
y_data = []

try:
    all_tags = d2v_model.dv.index_to_key
    vectors_ref = d2v_model.dv
except AttributeError:
    all_tags = list(d2v_model.docvecs.doctags.keys())
    vectors_ref = d2v_model.docvecs

print(f"Found {len(all_tags)} documents in the model.")
print("Extracting labels and vectors...")

for tag in all_tags:
    try:
        label_part = tag.rsplit("_", 1)[0]
        if not label_part:
            continue
        vector = vectors_ref[tag]
        X_data.append(vector)
        y_data.append(label_part)
    except Exception as e:
        continue

X_data = np.array(X_data)
y_data = np.array(y_data)

print(f"Data prepared. Samples: {len(X_data)}")

# ==========================================
# 3. Label Encoding 與 資料切分
# ==========================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_data)
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_  # 存起來等等畫圖用

print(f"Detected Classes ({num_classes}): {class_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_encoded, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
)


# ==========================================
# 4. PyTorch Dataset 與 Model
# ==========================================
class PTTDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


train_loader = DataLoader(
    PTTDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    PTTDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
)


class PTTClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PTTClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


model = PTTClassifier(d2v_model.vector_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. 訓練
# ==========================================
print("Start Training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# ==========================================
# 6. 評估與視覺化 (新增與修改部分)
# ==========================================
print("Start Evaluation & Plotting...")
model.eval()
all_preds = []
all_labels_test = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels_test.extend(labels.cpu().numpy())

# --- 1. 混淆矩陣 (Confusion Matrix) ---
cm = confusion_matrix(all_labels_test, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 2. Classification Report 視覺化 ---
# output_dict=True 會回傳字典格式，方便我們轉成 DataFrame
report_dict = classification_report(
    all_labels_test, all_preds, target_names=class_names, output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()

# 移除 accuracy, macro avg, weighted avg 這些統計列，只保留各個看板的成績
# 如果你想保留 average，可以註解掉下面這一行
report_df_filtered = report_df.iloc[:-3, :]

# 準備畫圖：分為兩個子圖
# 左邊畫 Precision, Recall, F1 (範圍 0-1)
# 右邊畫 Support (範圍可能是幾千幾萬)
fig, ax = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [3, 1]})

# 左圖：Metrics Heatmap
sns.heatmap(
    report_df_filtered[["precision", "recall", "f1-score"]],
    annot=True,
    cmap="RdYlGn",
    fmt=".2f",
    vmin=0,
    vmax=1,
    ax=ax[0],
)
ax[0].set_title("Classification Metrics (Precision, Recall, F1)")

# 右圖：Support Bar Chart
# 因為 Support 數量級不同，不適合放進 Heatmap，改用 Bar chart
sns.barplot(
    x=report_df_filtered["support"],
    y=report_df_filtered.index,
    ax=ax[1],
    color="skyblue",
)
ax[1].set_title("Support (Sample Count)")
ax[1].set_xlabel("Number of Samples")
ax[1].set_ylabel("")  # 右邊的 Y 軸標籤可以隱藏，因為跟左圖對齊

plt.tight_layout()
plt.show()

# 文字版報告還是印出來備查
print("\nClassification Report (Text):")
print(classification_report(all_labels_test, all_preds, target_names=class_names))
