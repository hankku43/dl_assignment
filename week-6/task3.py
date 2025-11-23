import torch

# 設置輸出選項，讓浮點數顯示更簡潔
torch.set_printoptions(precision=4)

def run_tensor_tasks():
    print("--- PyTorch 張量 (Tensor) 基礎操作 ---")

    # ----------------------------------------------------
    # 任務 1: 從 Python list 建立張量，並印出形狀和資料類型
    # ----------------------------------------------------
    print("\n[任務 1: 從 Python List 建立張量]")
    python_list = [[2, 3, 1], [5, -2, 1]]
    
    # 建立張量。預設的 dtype 通常是 torch.long (整數) 或 torch.float32 (浮點數)
    tensor_a = torch.tensor(python_list)
    
    print(f"原始 List: {python_list}")
    print(f"建立的張量:\n{tensor_a}")
    print(f"張量形狀 (Shape): {tensor_a.shape}")
    print(f"資料類型 (DType): {tensor_a.dtype}")
    # 注意：由於輸入是整數，PyTorch 預設會將其設為 torch.int64 (長整數)


    # ----------------------------------------------------
    # 任務 2: 建立一個 3x4x2 隨機浮點數張量 (0 ~ 1)
    # ----------------------------------------------------
    print("\n[任務 2: 建立 3x4x2 隨機浮點數張量 (0 ~ 1)]")
    
    # torch.rand(size) 建立在 [0, 1) 區間均勻分佈的隨機浮點數
    tensor_b = torch.rand(3, 4, 2)
    
    print(f"建立的張量:\n{tensor_b}")
    print(f"張量形狀 (Shape): {tensor_b.shape}")
    print(f"張量元素數量 (Numel): {tensor_b.numel()}")


    # ----------------------------------------------------
    # 任務 3: 建立一個 2x1x5 填滿 1 的張量
    # ----------------------------------------------------
    print("\n[任務 3: 建立 2x1x5 填滿 1 的張量]")
    
    # torch.ones(size) 建立填滿 1 的張量
    tensor_c = torch.ones(2, 1, 5)
    
    print(f"建立的張量:\n{tensor_c}")
    print(f"張量形狀 (Shape): {tensor_c.shape}")


    # ----------------------------------------------------
    # 任務 4: 張量矩陣乘法 (Matrix Multiplication)
    # ----------------------------------------------------
    print("\n[任務 4: 矩陣乘法 (Matrix Multiplication)]")
    
    # 矩陣乘法要求：第一個張量 (A) 的行數必須等於第二個張量 (B) 的列數。
    # Tensor A: 2x3
    A = torch.tensor([[1, 2, 4], [2, 1, 3]], dtype=torch.float32) 
    # Tensor B: 3x1 (註：原始題目給的 [[5], [2], [1]] 是一個 3x1 的矩陣)
    B = torch.tensor([[5], [2], [1]], dtype=torch.float32)
    
    # 使用 torch.matmul 或 @ 運算符進行矩陣乘法
    # 結果形狀將是 2x1
    matrix_product = torch.matmul(A, B)
    
    print(f"張量 A (2x3):\n{A}")
    print(f"張量 B (3x1):\n{B}")
    print(f"矩陣乘法結果 (A @ B, 形狀 2x1):\n{matrix_product}")
    # 計算驗證：
    # 第一行: (1*5 + 2*2 + 4*1) = 5 + 4 + 4 = 13
    # 第二行: (2*5 + 1*2 + 3*1) = 10 + 2 + 3 = 15


    # ----------------------------------------------------
    # 任務 5: 張量元素級乘法 (Element-wise Product)
    # ----------------------------------------------------
    print("\n[任務 5: 元素級乘法 (Element-wise Product)]")

    # 元素級乘法要求：兩個張量的形狀必須完全相同 (或可廣播)
    # Tensor C: 3x2
    C = torch.tensor([[1, 2], [2, 3], [-1, 3]], dtype=torch.float32)
    # Tensor D: 3x2
    D = torch.tensor([[5, 4], [2, 1], [1, -5]], dtype=torch.float32)
    
    # 使用 * 運算符或 torch.mul 進行元素級乘法
    element_wise_product = C * D
    
    print(f"張量 C (3x2):\n{C}")
    print(f"張量 D (3x2):\n{D}")
    print(f"元素級乘法結果 (C * D, 形狀 3x2):\n{element_wise_product}")


if __name__ == "__main__":
    run_tensor_tasks()