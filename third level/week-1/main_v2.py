import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from SquarePadResize import SquarePadResize as spr

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

batch_size = 32
image_target_size = 64
num_classes = 62

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        spr(target_size=image_target_size, pad_color=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

# 可自由指定模型檔案
MODEL_PATH = "./cifar_net.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------- CNN 模型 ----------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一層 Block：1 -> 64
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) # 通道數必須與 out_channels 對齊
        
        # 第二層 Block：64 -> 128
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 第三層 Block：128 -> 256 (強烈建議加這層來降維！)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        
        # 這是抵抗過擬合的救星，絕對不能註解掉
        self.dropout = nn.Dropout(0.5)

        # 計算 fc1 輸入維度: 
        # 64 -> pool(32) -> pool(16) -> pool(8)
        # 經過三次 Pool，圖片縮小到 8x8。參數從 6,700萬 暴降到 1,600萬，有效逼迫模型學習特徵而不是死背。
        fc_input_dim = 256 * 8 * 8  
        
        self.fc1 = nn.Linear(fc_input_dim, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 每次 Conv 都要經過 BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1) # 攤平
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)       # 隨機斷開 50% 神經元，強迫模型不依賴特定記憶
        x = self.fc2(x)
        return x
    
# ---------------- 訓練函數 ----------------
def train(model, model_path=MODEL_PATH, epochs=45):
    trainset = torchvision.datasets.ImageFolder(
        root="./handwriting/augmented_images/augmented_images1", transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=6
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# ---------------- 測試函數 ----------------
# def test(model, model_path=MODEL_PATH):
#     testset = torchvision.datasets.ImageFolder(
#         root="./handwriting/handwritten-english-characters-and-digits/combined_folder/test",
#         transform=transform,
#     )
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=batch_size, shuffle=False, num_workers=4
#     )
#     classes = testset.classes

#     # 嘗試安全載入模型
#     if os.path.exists(model_path):
#         checkpoint = torch.load(model_path)
#         model.load_state_dict(checkpoint, strict=True)
#         print(f"Loaded model from {model_path}")
#     else:
#         print(f"Model file {model_path} not found!")
#         return

#     model.eval()
#     correct = 0
#     total = 0
#     correct_pred = {classname: 0 for classname in classes}
#     total_pred = {classname: 0 for classname in classes}

#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predictions = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predictions == labels).sum().item()
#             for label, prediction in zip(labels, predictions):
#                 if label == prediction:
#                     correct_pred[classes[label]] += 1
#                 total_pred[classes[label]] += 1


#     print(f"Accuracy on test set: {100 * correct / total:.2f}%")
#     for classname, correct_count in correct_pred.items():
#         accuracy = 100 * float(correct_count) / total_pred[classname]
#         print(f"Class {classname:5s} accuracy: {accuracy:.1f}%")
def test(model, model_path=MODEL_PATH):
    testset = torchvision.datasets.ImageFolder(
        root="./handwriting/handwritten-english-characters-and-digits/combined_folder/test",
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=6
    )
    classes = testset.classes

    # 載入模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found!")
        return

    model.eval()

    all_preds = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

    # ---------------- 混淆矩陣 ----------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,  # 用類別名稱當 x 軸
        yticklabels=classes,  # 用類別名稱當 y 軸
    )  # annot=True 可以直接在格子裡印數字
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # 可以印單類別準確率
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Class {classname:5s} accuracy: {accuracy:.1f}%")


# ---------------- 主程式 ----------------
def main():
    model = Net().to(device)

    # 可以自由指定模型檔案測試
    MODEL_PATH = "./cifar_net8.pth"

    if os.path.exists(MODEL_PATH):
        test(model, MODEL_PATH)
    else:
        train(model, MODEL_PATH)
        test(model, MODEL_PATH)


if __name__ == "__main__":
    main()
