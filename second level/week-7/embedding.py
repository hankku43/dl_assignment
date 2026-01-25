import os
import json
import re
import numpy as np
import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

# --- 設定常數 ---
RAW_JSON = "traffic-law.json"
RAW_CSV = "traffic_law_articles.csv"
PROCESSED_CSV = "processed_traffic_law_articles.csv"
MODEL_PATH = "traffic_law_doc2vec.model"


def json2csv():
    """從 JSON 載入並進行初步清洗儲存"""
    print(f"📂 正在處理原始 JSON: {RAW_JSON}...")
    with open(RAW_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)["LawArticles"]

    df = pd.DataFrame(data)
    df["specific_id"] = df["ArticleNo"].str.extract(r"第\s*([\d\-]+)\s*條")
    df = df[df["ArticleType"] != "C"].copy()
    df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")


def preprocess_raw_data():
    """對原始 CSV 做分切與清洗"""
    df = pd.read_csv(RAW_CSV, dtype={"specific_id": str, "ArticleContent": str})

    # 1. 過濾刪除條文
    mask_deleted = df["ArticleContent"].str.contains("刪除", na=False) & (
        df["ArticleContent"].str.len() <= 5
    )
    df = df[~mask_deleted].copy()

    # 2. 僅做基本的標點與換行清洗，保持條文完整性交給 Tokenizer 處理
    df["ArticleContent"] = df["ArticleContent"].str.replace(r"[\s\r\n]", "", regex=True)

    # 3. 過濾空行
    df = df[df["ArticleContent"].str.len() > 0]

    return df[["specific_id", "ArticleContent"]]


def get_tagged_docs(csv_path):
    """讀取斷詞後的 CSV 並轉換為 TaggedDocument"""
    print(f"📖 載入斷詞資料: {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # 打亂順序有助於訓練
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return [
        # 使用 chunk_id 作為 Tag，確保每個片段向量唯一
        TaggedDocument(words=row["FilteredContent"].split(), tags=[row["chunk_id"]])
        for _, row in df.iterrows()
    ]


def train_model(tagged_data):
    """配置參數並訓練模型"""
    print("🚀 開始訓練 Doc2Vec...")
    model = Doc2Vec(
        vector_size=200,
        dm=0,
        dbow_words=1,
        window=5,
        min_count=1,
        workers=8,
        negative=5,
        epochs=100,
    )
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(MODEL_PATH)
    print(f"✅ 模型已儲存至: {MODEL_PATH}")
    return model


def evaluate_model(model, csv_path, sample_n=100):
    """執行 Self-Similarity 測試評估品質"""
    print(f"🧪 執行 Self-Similarity 測試 (樣本數={sample_n})...")
    df = pd.read_csv(csv_path, dtype=str)
    eval_df = df.sample(n=min(sample_n, len(df)), random_state=42)

    ranks = []
    for row in eval_df.itertuples():
        words = str(row.FilteredContent).split()
        if not words:
            continue

        # 推論向量並比對
        vec = model.infer_vector(words, epochs=50)
        sims = model.dv.most_similar([vec], topn=10)

        try:
            rank = [tag for tag, _ in sims].index(str(row.chunk_id))
            ranks.append(rank)
        except ValueError:
            ranks.append(999)

    ranks = np.array(ranks)
    print("-" * 30)
    print(f"🏆 Top 1 準確度: {np.mean(ranks == 0)*100:.2f}%")
    print(f"🥈 Top 2 準確度: {np.mean(ranks <= 1)*100:.2f}%")
    print(f"🥈 Top 5 準確度: {np.mean(ranks <= 4)*100:.2f}%")
    print("-" * 30)


if __name__ == "__main__":
    # 1. 確保原始對照表存在
    if not os.path.exists(RAW_CSV):
        print("⚙️ 偵測到缺少原始對照表，啟動json轉換...")
        json2csv()

    # 2. 確保斷詞後的 CSV 存在
    if not os.path.exists(PROCESSED_CSV):
        print("⚙️ 偵測到缺少斷詞檔，啟動 LawTokenizer...")
        from tokenizer import LawTokenizer

        tokenizer = LawTokenizer(device=0)
        df_raw = preprocess_raw_data()
        df_processed = tokenizer.tokenize_and_filter_df(df_raw)
        df_processed.to_csv(PROCESSED_CSV, index=False, encoding="utf-8-sig")

    # 3. 訓練
    docs = get_tagged_docs(PROCESSED_CSV)
    model = train_model(docs)

    # 4. 測試
    evaluate_model(model, PROCESSED_CSV)
