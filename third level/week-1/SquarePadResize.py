import torchvision.transforms as transforms
from PIL import Image

# --- 1. 自定義的轉換工具 ---
class SquarePadResize:
    def __init__(self, target_size=128, pad_color=255):
        """
        target_size: 輸出的圖片大小 (例如 64x64)
        pad_color: 填充顏色，白底黑字的話，背景要是白色 (255)
        """
        self.target_size = target_size
        self.pad_color = pad_color

    def __call__(self, img):
        # img 是 PIL Image 物件
        w, h = img.size
        
        # 1. 計算縮放比例，以長邊為準
        ratio = self.target_size / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # 2. 縮放圖片 (使用抗鋸齒縮放)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # 3. 建立一張全新的正方形畫布 (背景全白)
        new_img = Image.new("L", (self.target_size, self.target_size), self.pad_color)
        
        # 4. 把縮放後的圖貼在正中間
        # 計算貼上的座標 (置中)
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
