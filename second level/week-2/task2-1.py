from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
import pandas as pd
import glob
import os


def main():
    # 1. 初始化驅動
    print("Initializing drivers ...")
    ws_driver = CkipWordSegmenter(model="bert-base", device=0)
    pos_driver = CkipPosTagger(model="bert-base", device=0)
    print("Initializing drivers ... done")

    # 2. 抓取檔案列表
    file_list = glob.glob("cleaned_ptt_*.csv")
    print(f"Found files: {file_list}")

    # 3. 逐一處理每個檔案
    for file_path in file_list:
        try:
            print(f"\nProcessing file: {file_path} ...")

            # --- A. 讀取與解析檔名 ---
            df = pd.read_csv(file_path, encoding="utf-8-sig")

            # 從檔名解析出看版名稱 (例如 cleaned_ptt_Baseball.csv -> Baseball)
            filename = os.path.basename(file_path)
            board_name = filename.replace("cleaned_ptt_", "").replace(".csv", "")

            # --- B. 執行 Pipeline ---
            # 轉為字串 list
            text_list = df["Article Title"].astype(str).tolist()

            # 執行斷詞 (batch_size 請依顯卡記憶體調整)
            ws = ws_driver(text_list, use_delim=True, batch_size=64)
            pos = pos_driver(ws, use_delim=True, batch_size=64)

            # --- C. 過濾資料 ---
            filtered_corpus = []
            for sentence_ws, sentence_pos in zip(ws, pos):
                clean_words = filter_words(sentence_ws, sentence_pos)
                filtered_corpus.append(clean_words)

            # --- D. 建立 DataFrame (Word1, Word2...) ---
            # Pandas 會自動將 List 展開成欄位，長度不足的補 None
            result_df = pd.DataFrame(filtered_corpus)

            # 重新命名欄位為 Word1, Word2...
            num_cols = result_df.shape[1]
            result_df.columns = [f"Word{i+1}" for i in range(num_cols)]

            # --- E. 插入 Label 欄位 ---
            # 直接在第 0 欄插入 Label，值全部填入 board_name
            result_df.insert(0, "Label", board_name)

            # --- F. 分開存檔 ---
            output_filename = f"processed_{board_name}.csv"
            result_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
            print(f"-> Saved: {output_filename} (Shape: {result_df.shape})")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue



# --- 核心篩選邏輯函數 ---
def filter_words(sentence_ws, sentence_pos):
    """
    根據詞性和內容篩選詞彙
    去除: WHITESPACE, 介係詞, 連接詞, 代名詞, 是, 很、最,
    """

    res = []
    for word, p in zip(sentence_ws, sentence_pos):

        # 1. 去除 WHITESPACE (空白或換行)
        if p == "WHITESPACE" or word.strip() == "":
            continue

        # 2. 去除 介係詞 (Preposition) -> CKIP 標籤: P
        if p == "P":
            continue

        # 3. 去除 連接詞 (Conjunction) -> CKIP 標籤以 'C' 開頭 (Caa, Cbb...)
        if p.startswith("C"):
            continue

        # 4. 去除 代名詞 (Pronoun) -> CKIP 標籤: Nh
        if p == "Nh":
            continue

        # 5. 去除 "是" (Verb 'to be') -> CKIP 標籤: SHI
        if p == "SHI":
            continue

        # 6. 去除程度用詞 (Adverb 'very') -> CKIP 標籤: Dfa
        if p == "Dfa":
            continue

        # 7. 去除特定標點符號
        if p == "PARENTHESISCATEGORY" or p == "COLONCATEGORY" or p == "PAUSECATEGORY":
            continue

        # 如果通過以上所有檢查，就保留該詞
        res.append(word)

    return res


if __name__ == "__main__":
    main()
