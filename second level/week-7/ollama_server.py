import subprocess
import time
import psutil
import requests
import sys
from typing import Optional


class OllamaServer:
    def __init__(self, model: str = "gemma3:4b"):
        self.model = model
        self.process: Optional[subprocess.Popen] = None
        self.service_url = "http://localhost:11434"

    def is_running(self) -> bool:
        try:
            res = requests.get(self.service_url, timeout=2)
            return res.status_code == 200
        except:
            return False

    def start(self) -> bool:
        if self.is_running():
            return True
        print("🚀 啟動 Ollama...")
        self.process = subprocess.Popen(
            ["ollama", "serve"],
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        time.sleep(5)
        subprocess.run(["ollama", "pull", self.model])
        return self.is_running()

    def stop(self):
        if self.process:
            print("🛑 正在強制關閉 Ollama 及其所有子程序...")
            try:
                # 獲取父進程對象
                parent = psutil.Process(self.process.pid)
                # 獲取所有子進程
                children = parent.children(recursive=True)
                
                # 先殺掉所有子進程
                for child in children:
                    child.kill()
                
                # 最後殺掉父進程
                parent.kill()
                
                # 等待進程完全消失（防止殭屍進程）
                psutil.wait_procs([parent] + children, timeout=3)
                print("✅ 已完全清理。")
            except psutil.NoSuchProcess:
                print("⚠️ 進程已經不存在。")
            except Exception as e:
                print(f"❌ 關閉時發生錯誤: {e}")
            finally:
                self.process = None
