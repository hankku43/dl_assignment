import glob
import os
import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

TRAIN_MODEL = "ptt_titles_doc2vec_ver3.model"
TEST_MODEL = "ptt_titles_doc2vec_ver3.model"
Total_DATA = "test_shuffled_total_data.csv"

def main(test_mode):
    # 檢查模型與 CSV 狀態
    if os.path.exists(TRAIN_MODEL) and os.path.exists(Total_DATA):
        print(">>> 模型與總表 CSV 均已存在，跳過訓練步驟，直接進入測試。")

    elif os.path.exists(Total_DATA):
        print(">>> 總表 CSV 已存在，但模型不存在，將重新訓練模型。")
        tagged_docs = get_tagged_docs()  # 從 CSV 讀取
        run_train(tagged_docs)

    else:
        print(">>> 模型與總表 CSV 均不存在，將進行資料處理與模型訓練。")
        df = save_total_data()  # 合併並存檔
        tagged_docs = get_tagged_docs(df)
        run_train(tagged_docs)

    if test_mode == 1:
        # 執行similarity測試
        test_similarity()
    elif test_mode == 2:
        # 抽查失敗樣本
        test_similarity()
        debug_failures()


def save_total_data() -> pd.DataFrame:
    # 如果缺少其中一個，則執行原本的訓練流程
    print("開始執行資料處理與模型訓練流程...")

    # 找出所有處理後的 CSV 檔案
    file_list = glob.glob("test_processed_*.csv")

    if not file_list:
        print("找不到檔案，請確認路徑。")
        return

    print(f"Found files: {file_list}")

    all_dfs = []

    for file_path in file_list:
        print(f"Reading file: {file_path} ...")
        df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str)

        # 從檔名解析出標籤名稱
        file_name = os.path.basename(file_path)
        file_label = file_name.replace("processed_", "").replace(".csv", "")

        # 為每一列建立唯一標籤
        df["unique_id"] = [f"{file_label}_{i+1}" for i in range(len(df))]

        all_dfs.append(df)

    # 合併所有 DataFrame
    full_df = pd.concat(all_dfs, ignore_index=True)

    # 打亂資料順序
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("資料已完成打亂。")

    # 將打亂後的總表存檔，方便 test() 讀取以及未來分類使用
    merged_filename = Total_DATA
    full_df.to_csv(merged_filename, index=False, encoding="utf-8-sig")
    print(f"總表已儲存至: {merged_filename}")
    return full_df


def get_tagged_docs(df=None) -> TaggedDocument:

    if df is None:
        df = pd.read_csv(Total_DATA, dtype=str)

    # 建立 TaggedDocument 列表
    tagged_docs = []

    # 提取所有 Word 欄位名稱
    word_columns = [col for col in df.columns if col.startswith("Word")]

    # 逐列建立 TaggedDocument
    print("正在轉換為 TaggedDocument...")
    # 使用 itertuples 會比 iterrows 快很多 (適合 100 萬筆)
    # for _, row in df.iterrows():
    for row in df.itertuples(index=False):
        # 1. 提取字詞並過濾掉 NaN 或空字串
        words = [
            str(getattr(row, col))
            for col in word_columns
            if pd.notna(getattr(row, col))
            and str(getattr(row, col)).strip() != ""
            and str(getattr(row, col)).lower() != "nan"
        ]

        # 2. 取得我們剛剛建立的 unique_id
        tag = str(row.unique_id)

        # 3. 封裝成 TaggedDocument
        tagged_docs.append(TaggedDocument(words=words, tags=[tag]))

    return tagged_docs


def run_train(tagged_docs):
    print("開始訓練模型...")

    # 初始化模型
    model = Doc2Vec(
        vector_size=300,  # 保持 300，提供足夠空間
        window=5,  # 縮小窗口，適合短標題
        min_count=5,
        workers=8,
        dm=0,  # DBOW 對短文本更強健
        dbow_words=1,  # 在 DBOW 模式下同時訓練詞向量，這會大幅提升語義質量
        negative=5,  # 增加負採樣數量，有助於推開不相關的標題
        epochs=40,  # 增加訓練輪數
    )

    # 建立詞彙表
    model.build_vocab(tagged_docs)
    print(f"詞彙表建立完成，共 {len(model.wv)} 個獨特詞彙。")

    # 開始訓練
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

    # 儲存模型
    model.save(TRAIN_MODEL)
    print("模型訓練完成並已儲存！")


def test_similarity():
    # 載入模型
    if not os.path.exists(TEST_MODEL):
        print("找不到模型檔案，請先執行 main()")
        return
    model = Doc2Vec.load(TEST_MODEL)

    # 從剛剛存好的總表讀取前 1000 筆做測試
    merged_filename = Total_DATA
    if not os.path.exists(merged_filename):
        print("找不到總表 CSV，無法進行測試。")
        return

    print("載入測試樣本...")
    # 只讀取前 1000 筆，節省記憶體
    eval_df = pd.read_csv(merged_filename, nrows=1000, dtype=str)
    word_columns = [col for col in eval_df.columns if col.startswith("Word")]

    ranks = []
    sample_size = len(eval_df)
    print(f"開始進行 {sample_size} 筆樣本的 Self-Similarity 測試...")

    for row in eval_df.itertuples(index=False):
        words = [
            str(getattr(row, col))
            for col in word_columns
            if pd.notna(getattr(row, col))
            and str(getattr(row, col)).strip() != ""
            and str(getattr(row, col)).lower() != "nan"
        ]
        original_tag = str(row.unique_id)

        # 推論
        inferred_vector = model.infer_vector(words, epochs=20)

        # 尋找相似 (注意：這裡是在 100 萬筆的 index 裡找最像的)
        sims = model.dv.most_similar([inferred_vector], topn=10)

        try:
            rank = [docid for docid, sim in sims].index(original_tag)
            ranks.append(rank)
        except ValueError:
            ranks.append(999)

    ranks = np.array(ranks)
    top1_count = np.sum(ranks == 0)
    top2_count = np.sum(ranks <= 1)

    print("-" * 30)
    print(f"Self-Similarity (Top 1): {top1_count / sample_size * 100:.2f}%")
    print(f"Second Self-Similarity (Top 2): {top2_count / sample_size * 100:.2f}%")
    print("-" * 30)


def debug_failures(num_samples=10000, show_count=20):
    # 1. 載入模型與總表
    model = Doc2Vec.load(TEST_MODEL)
    df = pd.read_csv(Total_DATA, nrows=100000, dtype=str)

    # 建立一個快速查詢字典：unique_id -> 原始標題文字
    print("正在建立查詢字典...")
    word_cols = [c for c in df.columns if c.startswith("Word")]

    def get_title_string(row):
        words = [
            str(getattr(row, col))
            for col in word_cols
            if pd.notna(getattr(row, col))
            and str(getattr(row, col)).strip() != ""
            and str(getattr(row, col)).lower() != "nan"
        ]
        return " ".join(words)

    # 建立 ID 到標題文字的映射
    id_to_title = {
        row.unique_id: get_title_string(row) for row in df.itertuples(index=False)
    }

    # 2. 開始抽查
    failures = []
    eval_df = df.head(num_samples)  # 取前 10000 筆測試

    print(f"開始抽查 {num_samples} 筆中的失敗案例...")
    for row in eval_df.itertuples(index=False):
        current_id = row.unique_id
        current_title = id_to_title[current_id]

        # 提取單字列表進行推論
        words = current_title.split()
        inferred_vector = model.infer_vector(words, epochs=20)

        # 找最相似的第一名
        sims = model.dv.most_similar([inferred_vector], topn=1)
        top1_id, score = sims[0]

        if top1_id != current_id:
            # 如果第一名不是自己，紀錄下來
            # 嘗試從字典找回第一名的文字，如果不在前10萬筆內則顯示 ID
            top1_title = id_to_title.get(top1_id, "[ID不在快取中，僅顯示ID]")

            failures.append(
                {
                    "original_id": current_id,
                    "original_title": current_title,
                    "predicted_id": top1_id,
                    "predicted_title": top1_title,
                    "score": score,
                }
            )

        if len(failures) >= show_count:
            break

    # 3. 印出結果
    print("\n=== 失敗樣本抽查結果 ===")
    for i, f in enumerate(failures):
        print(f"案例 {i+1}:")
        print(f"  [原標題] ({f['original_id']}): {f['original_title']}")
        print(f"  [模型誤認] ({f['predicted_id']}): {f['predicted_title']}")
        print(f"  [相似度分數]: {f['score']:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main(test_mode=1)
