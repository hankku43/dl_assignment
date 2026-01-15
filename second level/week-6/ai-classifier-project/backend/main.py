import os
import csv
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Annotated

# 從隔壁的 processor.py 引入我們自定義的 AI 推論工具
from processor import TextInferencePipeline

# --- [系統設定：日誌紀錄] ---
# 當程式出錯時，我們會在伺服器終端機印出詳細錯誤，但不會把具體錯誤細節傳給前端使用者（安全性考量）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- [系統事件：生命週期管理] ---
# 這個部分負責處理伺服器「啟動時」與「關閉時」要執行的大型任務
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 伺服器啟動 (Startup):
    logger.info(">>> 正在初始化 AI 推論引擎 (載入大型模型)...")
    try:
        # 將 AI 模型載入記憶體，並存放在 app.state 中，讓所有 API 都能共享它
        app.state.pipeline = TextInferencePipeline()
        yield  # 伺服器正常運行期間會停在這裡
    finally:
        # 2. 伺服器關閉 (Shutdown):
        # 釋放記憶體資源
        if hasattr(app.state, "pipeline"):
            del app.state.pipeline
            logger.info(">>> AI 模型已從記憶體釋放。")


# 初始化 FastAPI 應用程式
app = FastAPI(lifespan=lifespan)


# --- [資料格式定義：Pydantic 模型] ---
# 這裡定義了前端傳過來的資料「必須」長什麼樣子，如果不符格式，FastAPI 會直接擋掉

# get不用Pydantic
# class TitleInput(BaseModel):
#     """預測請求的格式：前端傳來的 JSON 必須包含一個 title 欄位"""
#     # 使用 Field 限制字數：最少 1 字，最多 1000 字，防止惡意超長字串
#     title: str = Field(..., min_length=1, max_length=100)


class FeedbackInput(BaseModel):
    """回饋請求的格式：包含使用者選的標籤與原始標題"""
    label: str
    article_title: str


# --- [API 路由定義] ---

@app.get("/api/model/predict")
async def predict(request: Request, title: Annotated[str, Query(min_length=5, max_length=1000)]):
    """接收標題，回傳 AI 預測的主題標籤"""
    try:
        # 從 app.state 提取預先載入好的 AI 模型
        pipeline = request.app.state.pipeline
        # 執行辨識邏輯
        result = pipeline.predict(title)
        return {"prediction": result}
    except Exception as e:
        # 在伺服器端留下錯誤紀錄
        logger.error(f"預測過程中發生錯誤: {e}")
        # 給前端一個通用的錯誤提示，不洩漏系統內部路徑
        raise HTTPException(status_code=500, detail="AI 辨識失敗，請稍後再試")


@app.post("/api/model/feedback")
def save_feedback(data: FeedbackInput):
    """接收使用者校正的回饋，並寫入 CSV 檔案存檔"""
    # 檔案會存在 backend 資料夾下的 user_feedback.csv
    file_path = "user_feedback.csv"
    try:
        # 檢查檔案是否已存在且有內容 (決定是否要寫入標題列)
        file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
        
        # 使用 utf-8-sig 編碼，確保 Windows Excel 開啟 CSV 時不會出現亂碼
        with open(file_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["Label", "Article Title"])
            if not file_exists:
                writer.writeheader()  # 第一次建立檔案時寫入標題
            writer.writerow({"Label": data.label, "Article Title": data.article_title})
        return {"status": "success"}
    except Exception as e:
        logger.error(f"CSV 寫入錯誤: {e}")
        raise HTTPException(status_code=500, detail="無法儲存您的回饋")


# --- [靜態檔案託管：連接 Vue 前端] ---

# 找出前端編譯後的檔案路徑 (../frontend/dist)
frontend_dist_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)

# 系統啟動前檢查：如果找不到 dist，代表前端還沒跑 npm run build
if not os.path.exists(frontend_dist_path):
    logger.warning("警告: 找不到 dist 資料夾。請確保在前端執行過 npm run build")

# 掛載前端的靜態資源 (CSS, JavaScript, 圖片)
# 當瀏覽器請求 /assets/... 時，FastAPI 會去 dist/assets 找檔案
if os.path.exists(os.path.join(frontend_dist_path, "assets")):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(frontend_dist_path, "assets")),
        name="assets",
    )


# [萬用路由]：處理單頁式應用程式 (SPA) 的跳轉
# 這是為了讓 Vue Router 能夠運作。如果路徑不是 API，就通通回傳 index.html
@app.get("/{catchall:path}")
async def serve_frontend(catchall: str):
    # 安全檢查：如果是 API 請求但沒對應到路由，應該噴 404 而不是回傳網頁
    if catchall.startswith("api/"):
        raise HTTPException(status_code=404, detail="API 路由不存在")

    # 尋找 Vue 的入口 HTML
    index_file = os.path.join(frontend_dist_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)

    # 如果連 index.html 都沒找到，提示開發者需要編譯前端
    return {"message": "前端頁面尚未編譯，請先執行 npm run build"}


# 程式進入點
if __name__ == "__main__":
    import uvicorn
    # host 0.0.0.0 代表允許區域網路內的所有裝置連線
    uvicorn.run(app, host="0.0.0.0", port=8000)