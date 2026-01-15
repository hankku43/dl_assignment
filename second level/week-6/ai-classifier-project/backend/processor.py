import torch
import torch.nn as nn
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
from gensim.models.doc2vec import Doc2Vec
import numpy as np

# 1. 依照你的訓練代碼定義模型結構
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

class TextInferencePipeline:
    def __init__(self):
        # 依照 LabelEncoder 預設的字母排序
        self.themes = [
            "Baseball", "Boy-Girl", "C_Chat", "HatePolitics", 
            "Lifeismoney", "Military", "PC_Shopping", "Stock", "Tech_Job"
        ]
        
        # A. 初始化 CKIP (建議使用 CPU，若顯存夠大可改 device=0)
        print("Initializing CKIP drivers...")
        self.ws_driver = CkipWordSegmenter(model="bert-base", device=-1)
        self.pos_driver = CkipPosTagger(model="bert-base", device=-1)
        
        # B. 載入 Doc2Vec 模型 (維度 300)
        print("Loading Doc2Vec model...")
        self.d2v_model = Doc2Vec.load("models/ptt_titles_doc2vec_ver3.model")
        
        # C. 載入分類器
        print("Loading PyTorch Classifier...")
        num_classes = len(self.themes)
        self.classifier = PTTClassifier(input_dim=300, num_classes=num_classes)
        # 載入權重檔案
        self.classifier.load_state_dict(torch.load("models/ptt_classifier.pth", map_location=torch.device('cpu')))
        self.classifier.eval()

    def filter_words(self, sentence_ws, sentence_pos):
        """ 篩選邏輯 """
        res = []
        for word, p in zip(sentence_ws, sentence_pos):
            if p == "WHITESPACE" or word.strip() == "": continue
            if p == "P": continue
            if p.startswith("C"): continue
            if p == "Nh": continue
            if p == "SHI": continue
            if p == "Dfa": continue
            if p in ["PARENTHESISCATEGORY", "COLONCATEGORY", "PAUSECATEGORY"]: continue
            res.append(word)
        return res

    def predict(self, title: str):
        # 1. CKIP 斷詞與過濾
        ws = self.ws_driver([title], use_delim=True)
        pos = self.pos_driver(ws, use_delim=True)
        clean_words = self.filter_words(ws[0], pos[0])
        
        # 2. Doc2Vec 推論向量 (使用與你測試一致的 20 epochs)
        inferred_vector = self.d2v_model.infer_vector(clean_words, epochs=20)
        
        # 3. 轉為 Tensor 進行預測
        input_tensor = torch.FloatTensor(inferred_vector).unsqueeze(0)
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            _, predicted = torch.max(outputs, 1)
            
        return self.themes[predicted.item()]