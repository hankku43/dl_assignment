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

# ==========================================
# 類別定義 (放在函式外以便重複使用)
# ==========================================
class PTTDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

# ==========================================
# 主程式函式
# ==========================================
def main(test_mode=0):
    """
    執行 PTT 看板分類模型的訓練與評估。
    
    Args:
        test_mode (int): 控制輸出的詳細程度
            0: 僅列印文字版 Classification Report
            1: 加畫混淆矩陣 (Confusion Matrix)
            2: 加畫指標熱力圖與 Support 長條圖 (Full Visualization)
    """
    
    # 1. 設定參數
    MODEL_PATH = "ptt_titles_doc2vec_ver3.model"
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 20
    TEST_SIZE = 0.2
    
    # 檢查是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 資料載入
    print("Loading Doc2Vec model...")
    try:
        d2v_model = Doc2Vec.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    X_data = []
    y_data = []

    # 判斷 Gensim 版本
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
            # 解析 Tag 取得 Label
            label_part = tag.rsplit("_", 1)[0]
            if not label_part:
                continue
            
            vector = vectors_ref[tag]
            X_data.append(vector)
            y_data.append(label_part)
        except Exception:
            continue

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    print(f"Data prepared. Samples: {len(X_data)}")

    # 3. 前處理與切分
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_

    print(f"Detected Classes ({num_classes}): {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_encoded, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
    )

    # 4. 建立 DataLoader
    train_loader = DataLoader(
        PTTDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        PTTDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    # 5. 初始化模型
    model = PTTClassifier(d2v_model.vector_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. 訓練迴圈
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
        # 每 5 個 epoch 印一次進度，避免洗版
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    # 7. 評估 (Inference)
    print("Start Evaluation...")
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

    # ==========================================
    # 視覺化邏輯控制 (根據 test_mode)
    # ==========================================
    
    # 基礎：永遠印出文字報告
    print("\n" + "="*40)
    print("Classification Report (Text):")
    print("="*40)
    print(classification_report(all_labels_test, all_preds, target_names=class_names))

    # Level 1: 混淆矩陣
    if test_mode >= 1:
        print("Displaying Confusion Matrix...")
        cm = confusion_matrix(all_labels_test, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Level 2: 詳細指標視覺化
    if test_mode >= 2:
        print("Displaying Detailed Metrics Visualization...")
        report_dict = classification_report(
            all_labels_test, all_preds, target_names=class_names, output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        
        # 移除統計列
        report_df_filtered = report_df.iloc[:-3, :]

        fig, ax = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [3, 1]})

        # 左圖：Metrics Heatmap
        sns.heatmap(
            report_df_filtered[["precision", "recall", "f1-score"]],
            annot=True, cmap="RdYlGn", fmt=".2f", vmin=0, vmax=1, ax=ax[0]
        )
        ax[0].set_title("Classification Metrics (Precision, Recall, F1)")

        # 右圖：Support Bar Chart
        sns.barplot(
            x=report_df_filtered["support"],
            y=report_df_filtered.index,
            ax=ax[1], color="skyblue"
        )
        ax[1].set_title("Support (Sample Count)")
        ax[1].set_xlabel("Number of Samples")
        ax[1].set_ylabel("") 

        plt.tight_layout()
        plt.show()

# ==========================================
# 程式進入點範例
# ==========================================
if __name__ == "__main__":
    # 可以在這裡修改 test_mode 參數進行測試
    # test_mode = 0 : 只印文字
    # test_mode = 1 : 印文字 + 混淆矩陣
    # test_mode = 2 : 印文字 + 混淆矩陣 + 詳細指標圖
    
    main(test_mode=0)