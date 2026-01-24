import pandas as pd
import requests
from gensim.models.doc2vec import Doc2Vec
from ollama_server import OllamaServer
from tokenizer import LawTokenizer

def rag_search(query: str, raw_df: pd.DataFrame, model: Doc2Vec, tokenizer: LawTokenizer, top_k: int = 5):
    # 1. 使用相同的斷詞邏輯處理使用者的輸入
    processed_query = tokenizer.tokenize_query(query)
    query_words = processed_query.split()
    
    # 印出使用者經過tokenize後的查詢內容
    print(f"\n🔎 經過斷詞與過濾後的查詢內容: {query_words}")
    
    # 2. 向量推論
    query_vector = model.infer_vector(query_words, epochs=20)
    sims = model.dv.most_similar([query_vector], topn=top_k)
    
    results = []
    for specific_id, score in sims:
        if score < 0.4:  # 如果分數低於 0.4 就不採用
            continue
        # 從 raw_df (traffic_law_articles.csv) 找原始條文
        match_row = raw_df[raw_df["specific_id"] == specific_id]
        
        if not match_row.empty:
            content = match_row.iloc[0]["ArticleContent"]
            results.append({
                "id": specific_id,
                "text": f"{match_row.iloc[0]['ArticleNo']}: {content}",
                "score": score
            })
    return results

def main_rag():
    # 啟動 Ollama
    server = OllamaServer("gemma3:4b")
    if not server.start(): return

    # 載入模型與資料
    tokenizer = LawTokenizer(device=0)
    model = Doc2Vec.load("traffic_law_doc2vec.model")
    raw_df = pd.read_csv("traffic_law_articles.csv")

    try:
        while True:
            user_query = input("\n🔍 查詢 (或輸入 exit): ")
            if user_query.lower() == 'exit': break
            
            # 搜尋
            search_results = rag_search(user_query, raw_df, model, tokenizer)
            if not search_results:
                print("⚠️ 無相關法規可供參考。")
                continue
            
            # 組合 Context
            context = "\n".join([r['text'] for r in search_results])
            
            # 印出檢索到的法規
            print("\n📄 檢索到的相關法規：")
            print(context)

            # 呼叫 Ollama (回復你原本的 prompt 格式)
            prompt = f"""根據以下交通法規，回答用戶的問題：
法規內容：
{context}
用戶問題：{user_query}
如果法規內容沒有提到，請回答「無相關資訊，無法回答」。"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma3:4b", "prompt": prompt, "stream": False},
                timeout=60
            )
            print(f"\n🤖 回答：\n{response.json().get('response')}")
            
    finally:
        server.stop()

if __name__ == "__main__":
    main_rag()