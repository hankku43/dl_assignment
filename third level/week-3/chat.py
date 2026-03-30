import torch
from classical_chinese_lm import (
    CharTokenizer,
    ClassicalChineseLM,
    load_data,
    generate_text,
)


def main():
    # 1. 準備資料和 Tokenizer
    # 注意：這裡的 file_path 必須跟訓練時一模一樣，這樣 Tokenizer 的字元對應 ID 才會完全相同
    file_path = ".json"  # 這裡請替換成您實際的訓練資料路徑

    try:
        print("載入資料以重建 Tokenizer...")
        text = load_data(file_path)
    except FileNotFoundError:
        print(
            f"錯誤：找不到訓練資料 '{file_path}'，必須載入訓練資料才能還原 Tokenizer。"
        )
        return

    tokenizer = CharTokenizer(text)
    print(f"Tokenizer 重建成功，字典大小 (Vocab Size): {tokenizer.vocab_size}")

    # 2. 初始化模型結構 (需與訓練時的參數一模一樣)
    d_model = 256
    nhead = 8
    num_layers = 4
    dim_feedforward = 512
    dropout = 0.1

    model = ClassicalChineseLM(
        tokenizer.vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout
    )

    # 3. 判斷硬體與載入權重
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"使用運算裝置: {device}")

    model_path = "classical_chinese_lm.pth"
    try:
        # map_location=device 確保不管之前是用 GPU 還是 CPU 訓練的，都能正確載入到目前的裝置
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        print(f"成功載入模型權重: {model_path}")
    except FileNotFoundError:
        print(
            f"錯誤：找不到模型權重檔 '{model_path}'，請先執行 `classical_chinese_lm.py` 完成訓練。"
        )
        return

    # 測試模式，關閉 Dropout 等隨機機制
    model.eval()
    model.to(device)

    # 4. 開始終端機互動迴圈
    print("\n" + "=" * 50)
    print(" 古文生成語言模型 - 互動對話模式啟動")
    print(" 請輸入開頭文字（或輸入 'exit', 'q' 來離開程式）")
    print("=" * 50 + "\n")

    while True:
        try:
            # 接收使用者的輸入
            user_input = input("【您的提示詞】: ")
        except (KeyboardInterrupt, EOFError):
            # 捕捉 Ctrl+C 或是 Ctrl+Z 等中斷訊號
            print("\n程式結束。")
            break

        # 判斷是否要退出
        if user_input.strip().lower() in ["exit", "q", "quit"]:
            print("離開程式。")
            break

        # 如果沒有輸入內容，就重新要求輸入
        if not user_input.strip():
            continue

        # 進行生成文字
        try:
            sample = generate_text(
                model=model,
                tokenizer=tokenizer,
                start_str=user_input,
                max_len=100,  # 可在此調整生成的最長字數
                device=device,
                temperature=0.8,  # 可調整溫度 (較大偏向隨機，較小偏向保守)
            )
            print(f"【模型生成】:\n{sample}")
        except Exception as e:
            # 處理可能出現的錯誤（例如：輸入了訓練資料中從未出現過的字）
            print(f"生成過程發生錯誤: {e}")
            print("請嘗試使用其他常見的中文字。")

        print("-" * 50)


if __name__ == "__main__":
    main()
