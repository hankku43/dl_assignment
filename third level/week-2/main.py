# --- 引入必要的函式庫 ---
import torch  # PyTorch 深度學習框架核心
import torch.utils.data  # 提供 Dataset 和 DataLoader 等資料處理工具
import pandas as pd  # 用來讀取 CSV 檔案 (處理結構化資料最強的工具)
import os  # 用來處理檔案路徑 (如組合資料夾路徑與檔名)
from PIL import Image  # Python Imaging Library，用來開啟與處理圖片
import transforms as T_detector
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from engine import train_one_epoch, evaluate
import utils
from torchvision.models.detection.rpn import AnchorGenerator


# --- 定義資料集類別 ---
# 繼承 torch.utils.data.Dataset，這是 PyTorch 的標準規範
class VehicleDataset(torch.utils.data.Dataset):

    # 1. 初始化函式 (__init__)
    # 這個函式只會在建立 Dataset 物件時執行一次，用來設定基本參數
    def __init__(self, csv_file, image_dir, transforms=None):
        # 讀取包含標註資訊的 CSV 檔 (格式: filename, width, height, class, xmin, ymin, xmax, ymax)
        self.df = pd.read_csv(csv_file)
        self.grouped = self.df.groupby("filename")

        # 儲存圖片所在的資料夾路徑
        self.image_dir = image_dir

        # 儲存要進行的影像轉換 (如轉 Tensor、增強資料等)
        self.transforms = transforms

        # [關鍵邏輯] CSV 中每一列是一個框，但一張圖可能有多個框 (多列)。
        # 我們訓練是以「張」為單位，所以要取出所有唯一的檔名 (去重複)。
        self.image_ids = self.df["filename"].unique()

        # [類別映射]
        # 定義文字標籤轉數字 ID 的字典。
        # !!! 重要 !!!：在 PyTorch 物件偵測中，ID 0 永遠保留給「背景 (Background)」。
        # 所以你的實際物體 (Car, Truck...) 必須從 1 開始編號。
        self.class_to_id = {
            "Bus": 1,
            "Car": 2,
            "Motorcycle": 3,
            "Pickup": 4,
            "Truck": 5,
        }

    # 2. 獲取單筆資料函式 (__getitem__)
    # 這是最重要的地方！當 DataLoader 要拿第 idx 筆資料去訓練時，就會呼叫這個函式。
    def __getitem__(self, idx):
        # --- 步驟 A: 讀取圖片 ---
        # 從唯一檔名列表中，拿出第 idx 個檔名
        img_name = self.image_ids[idx]

        # 組合完整的圖片路徑 (例如: "images/frame_123.jpg")
        img_path = os.path.join(self.image_dir, img_name)

        # 使用 PIL 開啟圖片，並強制轉為 RGB 模式 (避免有些圖是 RGBA 或灰階導致錯誤)
        img = Image.open(img_path).convert("RGB")

        # --- 步驟 B: 獲取該圖片的所有標籤 (Bounding Boxes) ---
        # 從 CSV (DataFrame) 中，篩選出所有檔名等於 img_name 的資料列
        # 這樣我們就抓到了這張圖裡面所有的車子

        # 考慮時間複雜度，直接使用 groupby 的 get_group 方法會好。
        # records = self.df[self.df["filename"] == img_name]
        records = self.grouped.get_group(img_name)

        # 原本的寫法 (慢)
        # --------------------------------------------------------
        # 初始化兩個空串列，用來裝座標和類別 ID
        # boxes = []
        # labels = []
        # 逐列讀取 (也就是逐個框讀取)
        # for _, row in records.iterrows():
        # # 取出座標: 左上角(xmin, ymin) 和 右下角(xmax, ymax)
        # xmin = row["xmin"]
        # ymin = row["ymin"]
        # xmax = row["xmax"]
        # ymax = row["ymax"]
        # # 將座標加入列表
        # boxes.append([xmin, ymin, xmax, ymax])
        # # 取出文字類別 (如 'Car')，並透過字典轉成數字 ID (如 1)
        # label_name = row["class"]
        # labels.append(self.class_to_id[label_name])
        # --------------------------------------------------------

        # 修改後 (快)
        # 直接取出 numpy 陣列進行切片
        box_data = records[["xmin", "ymin", "xmax", "ymax"]].values
        boxes = box_data.tolist()

        label_data = records["class"].map(self.class_to_id).values
        labels = label_data.tolist()

        # --- 步驟 C: 轉換為 Tensor 格式 ---
        # 轉換 boxes 列表為 PyTorch Tensor，資料型態必須是 float32 (這是模型要求的)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # 轉換 labels 列表為 PyTorch Tensor，資料型態必須是 int64 (整數)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 記錄這是第幾張圖 (用 idx)，有時候除錯或評估會用到
        image_id = idx

        # 計算每個框的面積 (Area)
        # 公式: (xmax - xmin) * (ymax - ymin)
        # 這是為了 COCO 評估標準用的 (它會分別計算大、中、小物體的準確度)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # 定義 iscrowd (是否擁擠)。
        # 設為 0 代表這是單獨的物體；設為 1 代表這是一群無法區分的東西。
        # 一般簡單的物件偵測都設為 0 (全零向量)。
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        # --- 步驟 D: 包裝成 target 字典 ---
        # Faster R-CNN 模型要求輸入必須是一個字典 (Dict)，包含以下特定的 key
        target = {}
        target["boxes"] = boxes  # 所有框的座標 [N, 4]
        target["labels"] = labels  # 所有框的類別 [N]
        target["image_id"] = image_id  # 圖片編號
        target["area"] = area  # 框面積
        target["iscrowd"] = iscrowd  # 是否擁擠

        # --- 步驟 E: 資料增強與轉換 ---
        # 如果有設定 transforms (例如轉 Tensor、隨機翻轉)，就在這裡執行
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # 回傳圖片 (Tensor) 和對應的標籤資訊 (Dict)
        return img, target

    # 3. 長度函式 (__len__)
    # 告訴 DataLoader 我們總共有幾張圖片
    def __len__(self):
        return len(self.image_ids)


# --- 定義轉換工具 ---
# def get_transform(train):
#     transforms = []

#     # 1. 必備：將 PIL 圖片轉為 Tensor
#     # 注意：這一步轉出來的是 uint8 (0-255)，還不是模型要的 float
#     transforms.append(T_detector.PILToTensor())

#     # 2. 必備：轉為 float 並歸一化 (Scale)
#     # 這一行非常重要！它會把 [0, 255] 的整數轉成 [0.0, 1.0] 的浮點數
#     # 如果少了這行，模型會因為數值太大而無法收斂 (Loss 降不下來)
#     transforms.append(T_detector.ToDtype(torch.float, scale=True))

#     if train:
#         # 3. 訓練時加入隨機翻轉
#         # 你的檔案裡有寫好 RandomHorizontalFlip，它會自動翻轉 Bounding Boxes
#         transforms.append(T_detector.RandomHorizontalFlip(0.5))
#         transforms.append(T_detector.RandomPhotometricDistort())
#         transforms.append(T_detector.RandomIoUCrop())

#     return T_detector.Compose(transforms)
def get_transform(train):
    transforms = []
    
    # 1. 基礎轉換：轉成 Tensor
    transforms.append(T_detector.PILToTensor())
    
    if train:
        # 2. [優化] 多尺度訓練：隨機選擇短邊尺寸 (480~800)
        # 這樣模型會更適應遠近不同的車輛
        transforms.append(T_detector.RandomShortestSize(
            min_size=(480, 512, 544, 576, 600, 640, 672, 704, 736, 768, 800), 
            max_size=1333
        ))
        
        # 3. [優化] 光影增強：應對不同天氣與反光
        transforms.append(T_detector.RandomPhotometricDistort(p=0.5))
        
        # 4. [優化] 隨機縮小：幫助偵測小目標
        transforms.append(T_detector.RandomZoomOut(fill=[0, 0, 0], p=0.5))
        
        # 5. 基礎增強：水平翻轉
        transforms.append(T_detector.RandomHorizontalFlip(0.5))
        
    else:
        # 測試時，固定短邊為 800 即可，確保評估基準一致
        transforms.append(T_detector.RandomShortestSize(min_size=800, max_size=1333))

    # 最後必備：轉成 float 並歸一化
    transforms.append(T_detector.ToDtype(torch.float, scale=True))
    
    return T_detector.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    # 1. 載入預訓練模型 (COCO權重)
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0, 3.0),)
    )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        min_size=1200,
        max_size=1333,
        anchor_generator=anchor_generator,
    )

    # 2. 獲取分類器的輸入特徵數
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 3. 替換預訓練的頭部 (Head) 為新的，類別數為 num_classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def visualize_random_prediction(model, dataset, device):
    model.eval()

    # 隨機選 4 張不同圖片
    indices = random.sample(range(len(dataset)), 4)

    num_classes = len(dataset.class_to_id)

    colors = plt.cm.tab10.colors[:num_classes]
    id_to_class = {v: k for k, v in dataset.class_to_id.items()}

    # 建立 2x2 畫布
    _, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        img, _ = dataset[idx]

        with torch.no_grad():
            prediction = model([img.to(device)])[0]

        ax.imshow(img.permute(1, 2, 0).cpu().numpy())

        # 畫出預測框
        for i in range(len(prediction["boxes"])):
            score = prediction["scores"][i].item()
            if score > 0.5:
                box = prediction["boxes"][i].cpu().numpy()
                label_id = prediction["labels"][i].item()
                label_name = id_to_class.get(label_id, "Unknown")
                color = colors[label_id - 1]

                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                ax.text(
                    box[0],
                    box[1] - 5,
                    f"{label_name}: {score:.2f}",
                    color="white",
                    fontsize=10,
                    bbox=dict(facecolor=color, alpha=0.5),
                )

        ax.set_title(f"Image ID: {idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # --- 設定路徑 ---
    model_path = "vehicle_detection_model_5.pth"
    train_csv = "./vehicles-data/r-cnn-data/vehicles_images/train_labels.csv"
    test_csv = "./vehicles-data/r-cnn-data/vehicles_images/test_labels.csv"
    test_img_dir = "./vehicles-data/r-cnn-data/vehicles_images/test"
    train_img_dir = "./vehicles-data/r-cnn-data/vehicles_images/train"

    # --- 準備資料 (不論訓練或評估都需要用到 Test Set) ---
    dataset_test = VehicleDataset(test_csv, test_img_dir, get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        collate_fn=utils.collate_fn,
        pin_memory=True,  # 讓 Tensor 鎖定在記憶體中，加速傳輸到 GPU
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = 6
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # --- 關鍵邏輯：判斷模型是否存在 ---
    if os.path.exists(model_path):
        print(f"找到預訓練權重 {model_path}，直接進行評估與視覺化...")
        # 載入權重
        model.load_state_dict(torch.load(model_path, map_location=device))

        # 1. 做評估 (mAP)
        evaluate(model, data_loader_test, device=device)

        # 2. 隨機抓圖視覺化
        visualize_random_prediction(model, dataset_test, device)

    else:
        print(f"未找到權重檔，開始重新訓練...")
        # 建立訓練集
        dataset = VehicleDataset(train_csv, train_img_dir, get_transform(train=True))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=6,
            collate_fn=utils.collate_fn,
        )

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        num_epochs = 100
        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            lr_scheduler.step()
            evaluate(model, data_loader_test, device=device)

        # 訓練完記得儲存
        torch.save(model.state_dict(), model_path)
        print(f"訓練完成，模型已儲存至 {model_path}")

        # 訓練完也順便視覺化一張
        visualize_random_prediction(model, dataset_test, device)


if __name__ == "__main__":
    main()
