import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import random
import os

# 1. 設定要爬取的看板列表 (從你的網址中提取出的看板名稱)
# 注意：大小寫需與網址相符
BOARD_LIST = [
    "Boy-Girl",      # 男女版 (需 18+)
    "C_Chat",        # 希洽
    "HatePolitics",  # 政黑 (需 18+)
    "Lifeismoney",   # 省錢
    "Military",      # 軍武
    "PC_Shopping",   # 電蝦
    "Stock",         # 股版
    "Tech_Job"       # 科技業
]

# 設定 User-Agent 與 Cookies (關鍵：over18=1 通行證)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
COOKIES = {'over18': '1'}

MAX_TITLES_PER_BOARD = 200000

def get_soup(url):
    """發送請求並回傳 BeautifulSoup 物件 (包含 Cookie)"""
    try:
        # 這裡加入了 cookies 參數
        resp = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=10)
        if resp.status_code == 200:
            return BeautifulSoup(resp.text, "html.parser")
        else:
            print(f"  [Error] HTTP {resp.status_code}: {url}")
            return None
    except Exception as e:
        print(f"  [Fail] {e}")
        return None

def get_last_page_number(board):
    """取得該看板最新的頁碼"""
    url = f"https://www.ptt.cc/bbs/{board}/index.html"
    soup = get_soup(url)
    
    if not soup:
        return None
    
    paging_div = soup.find("div", class_="btn-group btn-group-paging")
    if not paging_div:
        return None
    
    prev_link = paging_div.find("a", string=re.compile("上頁"))
    if prev_link and "href" in prev_link.attrs:
        href = prev_link["href"]
        match = re.search(r"index(\d+)\.html", href)
        if match:
            return int(match.group(1)) + 1
    return 1

def crawl_board(board_name):
    """爬取單一看板的主要邏輯"""
    print(f"\n====== 開始爬取看板：{board_name} ======")
    
    # 取得最新頁碼
    last_page = get_last_page_number(board_name)
    if not last_page:
        print(f"無法取得 {board_name} 的頁數，跳過。")
        return

    print(f"  最新頁碼為: {last_page}，目標: 200,000 筆或爬完為止。")
    
    titles_data = []
    
    # 倒序爬取
    for page in range(last_page, 0, -1):
        if len(titles_data) >= MAX_TITLES_PER_BOARD:
            print(f"  [Stop] {board_name} 已收集滿 {MAX_TITLES_PER_BOARD} 筆。")
            break
            
        url = f"https://www.ptt.cc/bbs/{board_name}/index{page}.html"
        soup = get_soup(url)
        
        if not soup:
            continue
            
        # 解析標題
        rents = soup.find_all("div", class_="r-ent")
        for rent in rents:
            title_div = rent.find("div", class_="title")
            
            # 過濾刪除文，只抓有連結的標題
            if title_div and title_div.find("a"):
                title_text = title_div.find("a").get_text(strip=True)
                
                titles_data.append({
                    "Board Name": board_name,
                    "Article Title": title_text
                })
        
        # 每 50 頁顯示一次進度
        if page % 50 == 0:
            print(f"  進度: 第 {page} 頁 | 目前累積: {len(titles_data)} 筆")
            
        # 簡單延遲，避免被鎖 IP
        time.sleep(0.1) 

    # 該看板爬完後，直接存檔
    filename = f"ptt_{board_name}.csv"
    df = pd.DataFrame(titles_data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"====== {board_name} 完成！已儲存至 {filename} (共 {len(titles_data)} 筆) ======")

# --- 主程式執行區 ---
if __name__ == "__main__":
    start_time = time.time()
    
    for board in BOARD_LIST:
        crawl_board(board)
        
    end_time = time.time()
    print(f"\n全部任務完成！總耗時: {end_time - start_time:.2f} 秒")