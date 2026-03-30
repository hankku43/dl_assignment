import torch
import torch.nn as nn
import torch.optim as optim
import json
import math
import os


# ==========================================
# 1. Load Data (資料載入區)
# ==========================================
def load_data(filepath):
    """
    讀取 JSON 格式的古文資料 (如《論語》、《孟子》)。
    在此步驟中，我們會將原本結構化的段落，全部串接成一個超長的純文字字串。
    這是訓練語言模型最基本的資料準備方式。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = ""
    for chapter_data in data:
        # 依序取出每個章節中的 paragraphs 陣列
        paragraphs = chapter_data.get("paragraphs", [])
        for p in paragraphs:
            # 將每段文字加上換行符號後，接到總字串 text 的尾巴
            text += p + "\n"
    return text


# ==========================================
# 2. Tokenizer (字元級別的斷詞器)
# ==========================================
class CharTokenizer:
    """
    文字與數字之間的「翻譯字典」(Tokenizer)。
    神經網路無法直接看懂文字(String)，只能處理數字(Tensor)。
    因此我們需要建立一個映射表，賦予每個出現過的中文字一個專屬的「整數 ID」。
    """

    def __init__(self, text):
        # set(text) 會挑下文章中所有不重複的字元 (包含標點與換行字元)
        # sorted() 將它們固定排好順序，確保每次執行程式時對應的 ID 是一致的
        chars = sorted(list(set(text)))

        # 建立兩個字典：
        # char2idx: 用來將「字元」轉成「數字 ID」(例如: '子' -> 105)
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        # idx2char: 用來把模型預測出的「數字 ID」還原成「人類讀得懂的字元」(例如: 105 -> '子')
        self.idx2char = {i: ch for i, ch in enumerate(chars)}

        # vocab_size 代表我們總共有多少個不重複的單字(約 2000 多個)
        self.vocab_size = len(chars)

    def encode(self, s):
        """將輸入的字串，轉換為一連串的數字 ID (List of integers)"""
        return [self.char2idx[c] for c in s if c in self.char2idx]

    def decode(self, l):
        """將模型預測的一連串數字 ID，還原為字串"""
        return "".join([self.idx2char[i] for i in l])


# ==========================================
# 3. Positional Encoding (位置編碼層)
# ==========================================
class PositionalEncoding(nn.Module):
    """
    Transformer 模型的 Self-Attention 機制是「一次平行看完整句話」的。
    模型天生不會知道「子」在「曰」前面這件事(沒有 sequence 的概念)。
    這段程式碼透過加入(加法計算)基於 sin 和 cos 函數組合出的規律訊號，
    來賦予每一個單字專屬的「絕對位置座標」，讓網路不僅知道字義，還知道語序。
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Dropout 是為了防止過擬合 (Overfitting)，隨機切斷部分神經元連接
        self.dropout = nn.Dropout(p=dropout)

        # 建立一個全 0 矩陣，準備儲存位置座標 (長度最長為 5000 個字)
        pe = torch.zeros(max_len, d_model)

        # position 是一個單維度的張量：[0, 1, 2, ..., 4999]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 計算特徵縮放項 (div_term)，利用指數函數 (exp) 和對數 (log) 計算頻率
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 對偶數維度 (0, 2, 4...) 套用正弦函數 (sine)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 對奇數維度 (1, 3, 5...) 套用餘弦函數 (cosine)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 調整張量形狀以配合 Transformer: (max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # register_buffer 的好處是：這些是固定的數學值，不是要訓練的參數，
        # 但它會跟著模型一起被存到 GPU 記憶體裡面，並在儲存模型時被保存下來。
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x 的形狀: (seq_len, batch_size, d_model)"""
        # 將輸入 x (已經查表轉換好的單字特徵向量) 直接與位置編碼「相加」
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


# ==========================================
# 4. Language Model (基於 Transformer Decoder 的語言模型)
# ==========================================
class ClassicalChineseLM(nn.Module):
    """
    這是生成式語言模型(Autoregressive LM)的核心。
    雖然它使用了 `nn.TransformerEncoderLayer`，但搭配了強迫模型不能向後看的
    Causal Mask(因果遮罩)，其本質與行為就完全等同於一個 Decoder-only 的結構 (如 GPT 系列)。
    """

    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super(ClassicalChineseLM, self).__init__()
        self.model_type = "Transformer"
        self.d_model = d_model

        # 1. 詞嵌入層 (Embedding Layer)：將整數 ID (0~Vocab Size) 轉換成 d_model (256維) 的語義向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 2. 定義位置編碼
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. 定義神經網絡的主幹：Transformer 編碼層
        # nhead 代表擁有多個專注力頭(Multi-head Attention)，能從不同角度理解文字關係
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 4. 輸出預測層 (Linear)：把 256 維度的特徵拉長回字典數目大小 (vocab_size)，
        # 這個向量裡的分數越高，代表模型覺得下一個出現這個字的機率越大！
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """
        生成因果遮罩 (Causal Mask)。
        這是一個上三角形矩陣，將「未來的字」的注意力分數設定為負無窮 (-inf)。
        這確保模型在預測第 N 個字時，絕對無法「偷看」到第 N+1 個以後的未來資訊。
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src, src_mask):
        """
        前向傳播 (當您呼叫 model(x) 時，內部會執行的這段主程序)
        src shape: (seq_len, batch_size)  # 一串數字 ID
        """
        # 第一步：查表找出 Embedding 向量，並放大數值以配合位置編碼的大小
        src = self.embedding(src) * math.sqrt(self.d_model)
        # 第二步：打上專屬位置印記
        src = self.pos_encoder(src)
        # 第三步：送入神經網路進行深層的特徵萃取與上下文融合
        output = self.transformer_encoder(src, src_mask)
        # 第四步：對每一個字計算出它「下一個單字」的預測分數表
        output = self.fc_out(output)
        return output


# ==========================================
# 5. Dataset creation wrapper (資料批次取樣)
# ==========================================
def get_batch(data, seq_len, batch_size):
    """
    這是一個無限提供訓練題材的函數。
    由於全部古文檔案太長，我們每次只隨機切取 `batch_size` 句、長度為 `seq_len` 的文字。
    """
    # 決定這次抽籤的「隨機起點」(為了避免出界，最大起點為資料長度扣除擷取長度)
    ix = torch.randint(len(data) - seq_len, (batch_size,))

    # x：是模型的「輸入題材」(例如輸入：子、曰、學、而)
    x = torch.stack([data[i : i + seq_len] for i in ix])

    # y：是模型的「標準答案」(Target)。
    # 注意！y 的切片向右平移了 1 格！(對應答案：曰、學、而、時)
    # 這就是為什麼稱它為語言模型，因為永遠在預測下一個未知的字！
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])

    # 進行矩陣轉置以符合 PyTorch Transformer [長度, 批次] 的預設要求格式
    return x.t(), y.t()


# ==========================================
# 6. Training Function (模型訓練主迴圈)
# ==========================================
def train_model(
    model, data, tokenizer, epochs=10, batch_size=32, seq_len=64, lr=0.001, device="cpu"
):
    # 將模型搬移至運算裝置 (GPU 或 CPU)
    model.to(device)

    # 損失函數：CrossEntropyLoss 是專門用來計算「多選一選擇題(如字元選詞)」預測誤差的算法
    criterion = nn.CrossEntropyLoss()
    # 優化器：Adam 負責根據計算好的誤差(倒回溯算出梯度)，精確地去微調網路裡的權重參數
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 切換為訓練模式 (此時 Dropout 等隨機丟棄機制才會生效以防止過度死背答案)
    model.train()

    # 將單純的數字 List 轉換成 GPU 運算所需的張量
    data_tensor = torch.tensor(data, dtype=torch.long)

    # 計算每個訓練週期大約要跑幾次「抽樣」才算粗略看過整篇文章一遍
    num_batches = len(data_tensor) // (batch_size * seq_len)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(num_batches):
            # 取出這批次要訓練的題目 x 與答案 y
            x, y = get_batch(data_tensor, seq_len, batch_size)
            x, y = x.to(device), y.to(device)

            # 建立阻擋模型看未來的眼罩 (Causal Mask)
            src_mask = model.generate_square_subsequent_mask(seq_len).to(device)

            # 第一步：把這回合的誤差舊帳清空
            optimizer.zero_grad()

            # 第二步：將題目送進網路計算預測 (這會自動觸發 forward)
            output = model(x, src_mask)

            # 第三步：算出所有單字與標準答案之間差異 (Loss)
            # view(-1, ...) 是為了把 [seq_len, batch_size] 兩個維度攤平變成 1D，才能丟入損失函數
            loss = criterion(output.view(-1, tokenizer.vocab_size), y.reshape(-1))

            # 第四步：時光倒流，沿著神經網路回去計算每個參數該調大還是調小 (Backpropagation)
            loss.backward()

            # (重要！) 梯度裁剪：給這班狂飆的神經網路戴上牽繩。
            # 如果某次誤差算出來過大，這行會強制等比例縮小梯度長度(限制在0.5)，避免模型更新爆炸變 NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # 第五步：優化器正式跨出步伐，更新所有的權重參數！
            optimizer.step()

            total_loss += loss.item()

            # 每 50 個 Step 記錄一下當下學習的除錯狀態
            if i % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{num_batches}], Loss: {loss.item():.4f}"
                )

        # 總結印出這個世代(Epoch)總平均的 Loss (誤差越來越小，代表模型越來越聽懂古文了)
        print(
            f"--> Epoch [{epoch + 1}/{epochs}] Average Loss: {total_loss / num_batches:.4f}"
        )


# ==========================================
# 7. Generation Function (文本生成機制)
# ==========================================
def generate_text(
    model, tokenizer, start_str="子曰：", max_len=100, device="cpu", temperature=0.8
):
    """
    這是一支非常具代表性的生成函數 (Sampling with Temperature)。
    模型的「寫作」模式：給定開頭幾個字，讓網路自言自語般地「接龍」下去。
    """
    # 切換為測試模式 (關閉 Dropout 防止瞎猜)
    model.eval()
    model.to(device)

    # 將人類餵給它的前綴文字("子曰：")轉成電腦懂的數字陣列
    chars = tokenizer.encode(start_str)

    # 增加一個維度變成 (seq_len, batch_size=1)，準備送給模型處理
    src = torch.tensor(chars, dtype=torch.long).unsqueeze(1).to(device)

    # torch.no_grad() 告訴 PyTorch 我們現在只是在生成測試，請它不用浪費效能偷偷記憶那些拿來算倒回溯用的計算圖
    with torch.no_grad():
        for _ in range(max_len):
            # 產生符合當前字串長度的 Causal Mask
            src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)
            # 送進神經網絡預測
            output = model(src, src_mask)

            # (核心！) 我們在預測「下一個新字」時，只關心最後一個字衍伸出來的特徵預測！
            # output[-1]：指的是序列維度的最後一行。
            # 雖然只有抽取最後一維，但因為前面的 Transformer Encoder 分析過了，這裡其實蘊含了整句上文的知識濃縮！
            prob = output[-1, 0, :] / temperature

            # 利用 Softmax 函數，把帶正負號的不規則分數(Logits)，正規化變成 0~1 之間加總剛好等於100% 的機率分佈
            prob = torch.softmax(prob, dim=0)

            # 有了這 2000 多個字的機率轉輪，使用 multinomial 按機率進行隨機骰骰子(隨機抽樣)
            # 這避免了傳統 argmax 每次都挑死板的第一名，讓文章每次產出的靈活度大增
            next_char = torch.multinomial(prob, 1).item()

            # 將抽到的字元 ID 加入原本的陣列裡！
            chars.append(next_char)

            # 重新包裝成張量，準備作為進入下一輪預測的新基礎 (這就是 自回歸 Auto-Regressive 的真諦！)
            src = torch.tensor(chars, dtype=torch.long).unsqueeze(1).to(device)

            # 模型如果認為該句號或是段落結束，可能會生成特定的換行 token，提早中斷結束寫作
            if tokenizer.idx2char[next_char] == "\n":
                break

    # 將整串拼寫好的數字轉回人類文言文字串
    return tokenizer.decode(chars)


# ==========================================
# 主程式執行區塊 (Main Execution)
# ==========================================
if __name__ == "__main__":
    file_path = ".json"  # 當前目錄下的 data 檔案

    print("1. 載入並處理古文資料中...")
    text = load_data(file_path)
    print(f"   總字數: {len(text)}")

    tokenizer = CharTokenizer(text)
    print(f"   字典大小 (Vocab Size): {tokenizer.vocab_size}")

    encoded_text = tokenizer.encode(text)

    # --- 模型超參數設定 (教學用，可自行測試影響) ---
    d_model = 256  # 每個單字在模型中代表的向量維度(空間長度)
    nhead = 8  # Attention頭數：幫助網路尋找不同關聯(如: 文法、指代)
    num_layers = 4  # Transformer深度：堆疊越深層智商越高，但越難訓練
    dim_feedforward = 512  # 多層感知機(MLP)的寬度
    dropout = 0.1  # 防死背(過擬合)機率

    model = ClassicalChineseLM(
        tokenizer.vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout
    )

    # 智慧判斷是否支援更快的運算晶片 (NVIDIA CUDA 或是 Apple M-Series MPS)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"2. 使用運算裝置: {device}")

    # 訓練超參數設定
    epochs = 10  # 整本書被翻了幾遍
    batch_size = 64  # 每批吸收多少句
    seq_len = 64  # 眼睛每次看到多長的字

    print("3. 開始訓練語言模型...")
    # 開始啟動訓練迴圈
    train_model(
        model,
        encoded_text,
        tokenizer,
        epochs=epochs,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )

    print("\n4. 訓練完成，進行文本生成測試...")
    # 給定不同提示字串 (Prompt) 查看模型即興發揮
    prompts = ["子曰：", "孟子見梁惠王", "天下之大"]
    for prompt in prompts:
        print("--------------------------------------------------")
        print(f"【輸入】: {prompt}")
        # temperature (溫度) 代表骰子的均勻度，越高越隨機破天荒，越低越保守中板
        sample = generate_text(
            model,
            tokenizer,
            start_str=prompt,
            max_len=100,
            device=device,
            temperature=0.8,
        )
        print(f"【生成】: {sample}")
    print("--------------------------------------------------")

    # 將辛辛苦苦訓練好的權重「大腦」保存下來，下次就能直接載入不重練
    save_path = "classical_chinese_lm.pth"
    torch.save(model.state_dict(), save_path)
    print(f"5. 模型權重已儲存至 {save_path}")
