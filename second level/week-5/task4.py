import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import Dataset, DataLoader
from gensim.models import Doc2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score


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
def main():
    """
    執行 PTT 看板分類模型的訓練與評估。
    """

    # 1. 設定參數
    # 取得目前這個 script (.py) 所在的絕對路徑
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 組合出模型的完整路徑
    MODEL_PATH = os.path.join(BASE_DIR, "ptt_titles_doc2vec_ver3.model")

    print(f"Looking for model at: {MODEL_PATH}")  # 印出來檢查路徑對不對
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
            print(
                f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}"
            )

   # 7. 評估與簡單答對率報告
    print("\n" + "="*50)
    print(f"{'看板名稱':<15} | {'總篇數':<8} | {'正確篇數':<8} | {'答對率'}")
    print("-" * 50)
    
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

    # 計算混淆矩陣
    cm = confusion_matrix(all_labels_test, all_preds)
    
    # 逐一計算每個類別的準確率
    for i, board_name in enumerate(class_names):
        correct_count = cm[i, i]            # 矩陣對角線即為預測正確的數量
        total_count = np.sum(cm[i, :])      # 該列的總和即為該看板的測試總篇數
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"{board_name:<15} | {total_count:<10} | {correct_count:<10} | {accuracy:.2f}%")

    # 顯示總體答對率
    overall_acc = accuracy_score(all_labels_test, all_preds) * 100
    print("-" * 50)
    print(f"總計所有看板平均答對率: {overall_acc:.2f}%")
    print("="*50)


# ==========================================
# 程式進入點範例
# ==========================================
if __name__ == "__main__":

    main()
