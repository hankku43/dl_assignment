import itertools
import urllib.request as req
import json
import datetime
import re
import csv

try:
    #抓取資料
    page = 1
    id_and_price_list = []
    high_rating_ids = []
    i5_prices = []
    pattern = re.compile(r"(?<= |-)i5(?![A-Za-z0-9])")  # i5前面一定是符號-或是空格，後面不是數字和英文字 #re.compile提前編譯，提高效能
        
    for page in itertools.count(1): #語意上比while True更明確
        with req.urlopen(f"https://ecshweb.pchome.com.tw/search/v4.3/all/results?cateid=DSAA31&pageCount=100&page={page}") as resp:
            data = json.load(resp)            
        prods = data["Prods"]
        if(not prods): break # 不存在頁數則prods為空字串，表示資料讀取完畢，跳出迴圈。        
        for prod in prods:
            
            #紀錄ID和價錢清單
            id_and_price_list.append({"ProductID":prod["Id"],"Price":prod["Price"]})
            
            
            #記錄高分數ID清單
            if(prod["reviewCount"] and prod["ratingValue"]>4.9):
                high_rating_ids.append(prod["Id"])
                
            #紀錄i5電腦價錢                
            if pattern.search(prod["Describe"]):
                i5_prices.append(prod["Price"])             
    
    #輸出檔案
    with open("products.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(obj["ProductID"] for obj in id_and_price_list))
            
    with open("best-products.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(id for id in high_rating_ids))
            
    #計算i5電腦平均價錢
    if i5_prices:
        avg = sum(i5_prices) / len(i5_prices)
        print(avg)
    else:
        print("查無i5電腦，無法計算價格平均值")
        
        
    #計算並輸出z-score檔案
    if id_and_price_list:
        mean = sum(obj["Price"] for obj in id_and_price_list) / len(id_and_price_list)
        sd = (sum((obj["Price"] - mean) ** 2 for obj in id_and_price_list) / len(id_and_price_list)) ** 0.5   
        
        for obj in id_and_price_list:
            obj["PriceZScore"] = (obj["Price"]-mean)/sd
            
        with open("standardization.csv", mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=id_and_price_list[0].keys())
            writer.writeheader()  # 寫入欄位名稱
            writer.writerows(id_and_price_list)  # 寫入資料
        
    else:
        raise Exception("獲取價格清單失敗，無法匯出standardization.csv")

#錯誤處理 KeyError
except KeyError as e:
    msg = f"[{datetime.datetime.now()}] resp中查無欄位：{e}\n"
    print("回應格式有錯：", msg)

    # 寫入錯誤紀錄檔
    with open("errorLog.txt", "a", encoding="utf-8") as f:
        f.write(msg)

except Exception as e:
    msg = f"[{datetime.datetime.now()}] 未知錯誤：{e}\n"
    print("發生未知錯誤：", msg)
    with open("errorLog.txt", "a", encoding="utf-8") as f:
        f.write(msg)