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
        self.stop_pos = {"Caa", "D", "DE", "Neu", "Nf", "Ng", "V_2", "PARENTHESISCATEGORY"}
        self.custom_stopwords = {"本", "條例", "者", "處", "新臺幣", "罰鍰", "下列", "情形", "之", "及", "或", "其"}

    def filter_words(self, sentence_ws, sentence_pos):
        filtered = []
        for word, pos in zip(sentence_ws, sentence_pos):
            if pos in self.stop_pos: continue
            if word in self.custom_stopwords: continue
            if len(word.strip()) <= 1 and not word.isalpha() and not word.isdigit(): continue
            if word.isdigit(): continue
            filtered.append(word)
        return filtered

    def tokenize_query(self, text: str) -> str:
        """供 RAG 查詢使用"""
        text = re.sub(r"[\s\r\n，。；：？！、]", "", text)
        ws = self.ws_driver([text], use_delim=True)
        pos = self.pos_driver(ws, use_delim=True)
        clean_words = self.filter_words(ws[0], pos[0])
        return " ".join(clean_words)

    def tokenize_and_filter_df(self, df):
        """供批次訓練使用"""
        print("⚙️ 開始斷詞與過濾...")
        text_list = df["ArticleContent"].astype(str).tolist()
        ws = self.ws_driver(text_list, use_delim=True, batch_size=64)
        pos = self.pos_driver(ws, use_delim=True, batch_size=64)
        
        filtered_corpus = []
        for s_ws, s_pos in zip(ws, pos):
            clean_words = self.filter_words(s_ws, s_pos)
            filtered_corpus.append(" ".join(clean_words)) 
        
        df["FilteredContent"] = filtered_corpus
        return df.drop(columns=["ArticleContent"])

