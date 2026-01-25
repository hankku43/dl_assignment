import re
import pandas as pd
import requests
from gensim.models.doc2vec import Doc2Vec
from ollama_server import OllamaServer
from tokenizer import LawTokenizer

is_debug: bool = True


def rag_search(
    query: str,
    raw_df: pd.DataFrame,
    model: Doc2Vec,
    tokenizer: LawTokenizer,
    top_k: int = 5,
    score_filter: float = 0.7,
    is_first: bool = False,
):
    if is_debug:
        print("\n[DEBUG] 執行 RAG 搜尋...")
        print(f"[DEBUG] 原始查詢內容: {query}")
        print(f"[DEBUG] score_filter = {score_filter}")

    # 1. 使用相同的斷詞邏輯處理使用者的輸入
    processed_query = tokenizer.tokenize_query(query)
    query_words = processed_query.split()

    # 印出使用者經過tokenize後的查詢內容
    print(f"\n🔎 經過斷詞與過濾後的查詢內容: {query_words}")

    # 2. 向量推論
    query_vector = model.infer_vector(query_words, epochs=20)
    sims = model.dv.most_similar([query_vector], topn=top_k)
    seen_articles = set()  # 避免重複檢索到同一個條文的不同片段

    results = []
    for chunk_id, score in sims:
        if score < (0.2 if is_first else score_filter):
            # 如果分數低於score_filter就不採用
            continue
        # 從 chunk_id 提取原始 specific_id (把 _0 去掉)
        specific_id = chunk_id.split("_")[0]
        if specific_id in seen_articles:
            continue
        
        # 從 raw_df (traffic_law_articles.csv) 找原始條文
        match_row = raw_df[raw_df["specific_id"] == specific_id]

        if not match_row.empty:
            content = match_row.iloc[0]["ArticleContent"]
            results.append(
                {
                    "id": specific_id,
                    "text": f"{match_row.iloc[0]['ArticleNo']}: {content}",
                    "score": score,
                }
            )

    if is_first:
        # 如果是第一次搜尋且沒有結果，拋出錯誤
        if not results:
            raise ValueError("查無相關法規，請嘗試更改查詢內容。")
        else:
            # 去除分數低於score_filter的結果
            results = [res for res in results if res["score"] >= score_filter]
            
    if is_debug:
        print(f"[DEBUG] 搜尋結果數量: {len(results)}")
        for res in results:
            print(f"[DEBUG] ---分數: {res['score']:.4f}, 內容: {res['text'][:30]}...")
    return results, query_words


def call_ollama_api(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gemma3:4b", "prompt": prompt, "stream": False},
        timeout=60,
    )
    return response.json().get("response")

def main_rag():
    server = OllamaServer("gemma3:4b")
    if not server.start(): return

    tokenizer = LawTokenizer(device=0)
    model = Doc2Vec.load("traffic_law_doc2vec.model")
    raw_df = pd.read_csv("traffic_law_articles.csv")

    try:
        while True:
            user_query = input("\n🔍 查詢 (或輸入 exit): ")
            if user_query.lower() == "exit": break

            # 執行重試管道
            final_response = run_rag_pipeline(user_query, raw_df, model, tokenizer)
            print(f"\n🤖 回答：\n{final_response}")
    finally:
        server.stop()

def clean_llm_output(text):
    """徹底清洗 LLM 回傳的關鍵字，只留下中文字與空格"""
    # 移除底線、特殊符號、英文、數字（除非你法條有數字需求）
    text = re.sub(r"[A-Za-z0-9\!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?？＿]+", " ", text)
    # 移除多餘空格與換行
    words = text.replace("\n", " ").split()
    return " ".join(words)

def run_rag_pipeline(user_query, raw_df, model, tokenizer):
    """
    RAG 核心流程管道：包含「快速攔截」、「結果清洗」與「策略重試」
    """
    try:
        if is_debug:
            print("\n[DEBUG] 開始 RAG 查詢流程...")
        # --- 1. 第一次嘗試 (直接搜尋) ---
        if is_debug:
            print("\n[DEBUG] 第一次嘗試：直接搜尋...")
        search_results, query_words = rag_search(
            user_query, raw_df, model, tokenizer, 
            score_filter=0.8, is_first=True
        )
        
        # 嘗試產生回答
        response = generate_and_check(user_query, search_results)
        if is_debug:
            print(f"[DEBUG] 第一次嘗試回答結果: {response}")
        if response: 
            return response
            
    except ValueError as e:
        # 快速攔截：如果是完全無關的問題 (score < 0.2)，直接報錯結束
        return f"⚠️ {str(e)}"

    # --- 2. 進入重試管道 (當第一步無結果或模型無法回答時) ---
    if is_debug:
        print("\n[DEBUG] 進入重試管道...")  
    strategies = [
        ("同義詞擴充", 0.6, lambda qw: call_ollama_api(
            f"請將下列單詞分別擴充成3個法律同義詞，例如：'手機'->'行動電話 手持裝置'。單詞：{qw}。僅回傳詞彙，用空白分隔，不要有標點或說明。"
        )),
        ("法律用語轉換", 0.5, lambda qw: call_ollama_api(
            f"請將此口語轉換為正式法律檢索詞，例如：'車禍'->'交通事故 碰撞'。內容：{qw}。僅回傳詞彙，不要底線、不要說明。"
        )),
        ("HyDE (虛擬回答搜尋)", 0.4, lambda qw: call_ollama_api(
            f"請根據交通法規知識，簡單回答此問題：{user_query}。回答需包含正式法規術語。"
        ))
    ]

    for name, score, transform_func in strategies:
        if is_debug: print(f"\n[DEBUG] 嘗試策略：[{name}]...")
        
        # 決定轉換對象
        target = user_query if "HyDE" in name else query_words
        
        # 呼叫 LLM 取得擴充內容
        raw_llm_output = transform_func(target)
        
        # --- 關鍵修正：徹底清洗 LLM 的回傳內容 ---
        refined_keywords = clean_llm_output(raw_llm_output)
        
        # 為了保險，將「原始斷詞」與「LLM擴充詞」組合，強化核心語義
        # 這樣就算 LLM 給出奇怪的詞，原本的關鍵字（如：手機）也還會在
        original_keywords = " ".join(query_words)
        new_query = f"{original_keywords} {refined_keywords}"
        
        if is_debug: 
            print(f"[DEBUG] LLM 原始回傳: {raw_llm_output.strip()}")
            print(f"[DEBUG] 清洗合併後內容: {new_query}")

        # 如果洗完變空的，就跳過此策略
        if not refined_keywords.strip():
            continue

        # 執行搜尋 (is_first=False)
        search_results, _ = rag_search(new_query, raw_df, model, tokenizer, score_filter=score, is_first=False)
        
        # 產生並檢查回答 (如果 LLM 說無法回答，會回傳 None 並繼續下一個策略)
        is_hyde = "HyDE" in name
        response = generate_and_check(user_query, search_results, fake_answer=new_query if is_hyde else None)
            
        if response: 
            return response

    return "⚠️ 抱歉，目前找不到相關法規規定，請嘗試更換關鍵字查詢。"


def generate_and_check(user_query, search_results, fake_answer=None):
    """
    組合 Prompt 並檢查模型是否給出實質回應
    """
    if not search_results:
        return None

    context = "\n".join([f"分數: {r['score']}, 法規: {r['text']}" for r in search_results])
    
    if fake_answer:
        prompt = f"這是一則錯誤訊息：{fake_answer}。請根據以下法規更正：\n{context}\n回答原問題：{user_query}"
    else:
        prompt = f"法規內容：\n{context}\n根據上述內容回答問題：{user_query}\n若無相關資訊請回答「無法回答」。"

    response = call_ollama_api(prompt)
    
    # 如果模型回答包含「無法回答」，我們視為搜尋無效，回傳 None 觸發下一個策略
    if "無法回答" in response or "無相關資訊" in response:
        if is_debug:
            print("[DEBUG] 模型回應表示無法回答，進入下一策略...")
        return None
    
    return response

if __name__ == "__main__":
    main_rag()
