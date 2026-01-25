import json
import os
import pandas as pd
import re
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger


class LawTokenizer:
    def __init__(self, device=0):
        print("📂 正在載入 CKIP 模型...")
        self.ws_driver = CkipWordSegmenter(model="bert-base", device=device)
        self.pos_driver = CkipPosTagger(model="bert-base", device=device)
        # 完整保留你設定的詞性與停用詞
        self.stop_pos = {"Caa", "D", "DE", "Neu", "Ng", "V_2", "PARENTHESISCATEGORY"}
        self.custom_stopwords = {
            "本",
            "條例",
            "者",
            "處",
            "新臺幣",
            "罰鍰",
            "下列",
            "情形",
            "之",
            "及",
            "或",
            "其",
            "違反",
            "規定",
            "處罰",
            "其他",
            "元",
            "條",
            "項",
            "款",
            "依",
            "最",
            "前",
            "並"
        }

    def filter_words(self, sentence_ws, sentence_pos):
        filtered = []
        for word, pos in zip(sentence_ws, sentence_pos):
            if pos in self.stop_pos:
                continue
            if word in self.custom_stopwords:
                continue
            if len(word.strip()) <= 1 and not word.isalpha() and not word.isdigit():
                continue
            if word.isdigit():
                continue
            filtered.append(word)
        return filtered

    def tokenize_query(self, text: str) -> str:
        """供 RAG 查詢使用"""
        text = re.sub(r"[\s\r\n，。；：？！、違法嗎罰]", "", text)
        ws = self.ws_driver([text], use_delim=True)
        pos = self.pos_driver(ws, use_delim=True)
        clean_words = self.filter_words(ws[0], pos[0])
        return " ".join(clean_words)

    def chunk_tokens(self, tokens, max_tokens=10, buffer_tokens=5):
        """
        將 token 列表依照指定數量分塊
        max_tokens: 每組 10 個詞
        buffer_tokens: 剩餘不足 5 個詞則合併
        """
        if len(tokens) == 0:
            return []
        
        # 如果總詞數本來就小於等於 15 (10+5)，直接回傳不切分
        if len(tokens) <= max_tokens + buffer_tokens:
            return [" ".join(tokens)]

        chunks = []
        i = 0
        while i < len(tokens):
            # 取出目前的區塊
            chunk = tokens[i : i + max_tokens]
            
            # 檢查剩下的詞數
            remaining = tokens[i + max_tokens :]
            
            # 如果剩下的詞數小於等於 buffer_tokens (5)，且不為 0
            # 則將剩下的詞全部併入目前這一組，並結束迴圈
            if 0 < len(remaining) <= buffer_tokens:
                chunk.extend(remaining)
                chunks.append(" ".join(chunk))
                break
            else:
                chunks.append(" ".join(chunk))
                i += max_tokens
        
        return chunks

    def tokenize_and_filter_df(self, df):
        print(f"⚙️ 開始斷詞與唯一分切... (處理行數: {len(df)})")
        text_list = df["ArticleContent"].astype(str).tolist()
        ws = self.ws_driver(text_list, batch_size=64)
        pos = self.pos_driver(ws, batch_size=64)

        all_chunks = []
        all_ids = []        # 原始 ID (例如 3)
        chunk_unique_ids = [] # 唯一標籤 (例如 3_0, 3_1)

        for specific_id, s_ws, s_pos in zip(df["specific_id"], ws, pos):
            clean_tokens = self.filter_words(s_ws, s_pos)
            chunks = self.chunk_tokens(clean_tokens, max_tokens=25, buffer_tokens=10)
            
            for i, c in enumerate(chunks):
                all_chunks.append(c)
                all_ids.append(specific_id)
                # 產生唯一標籤：原始ID + 底線 + 序號
                chunk_unique_ids.append(f"{specific_id}_{i}")

        new_df = pd.DataFrame({
            "chunk_id": chunk_unique_ids, # 用這個當 Tag 訓練
            "specific_id": all_ids,      # 保留這個用來指回原條文
            "FilteredContent": all_chunks
        })

        # 過濾太短的片段
        new_df = new_df[new_df["FilteredContent"].str.split().str.len() >= 2]
        return new_df
