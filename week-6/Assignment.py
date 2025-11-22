import json
import math
import csv
import random
import re
from typing import Sequence, List, Dict


class Network:

    EPSILON = 1e-15

    # ================================================================
    #  一、Activation Functions
    # ================================================================
    class Activation:
        """定義常見激勵函數"""

        @staticmethod
        def linear(x: Sequence[float]) -> List[float]:
            return list(x)

        @staticmethod
        def linear_derivative(x: float) -> float:
            return 1

        @staticmethod
        def relu(x: Sequence[float]) -> List[float]:
            return [max(0, v) for v in x]

        @staticmethod
        def relu_derivative(x: float) -> float:
            return 1 if x > 0 else 0

        @staticmethod
        def sigmoid(x: Sequence[float]) -> List[float]:
            return [1 / (1 + math.exp(-v)) for v in x]

        @staticmethod
        def sigmoid_derivative(node_output: float) -> float:
            return node_output * (1 - node_output)

        @staticmethod
        def softmax(x: Sequence[float]) -> List[float]:
            max_x = max(x)
            exp_x = [math.exp(v - max_x) for v in x]
            total = sum(exp_x)
            return [v / total for v in exp_x]

    # ================================================================
    # 二、Loss Functions
    # ================================================================
    class Loss:
        """定義常見損失函數"""

        @staticmethod
        def mse(expects, outputs):
            """Mean Squared Error"""
            expects = Network._to_list(expects)
            outputs = Network._to_list(outputs)
            n = len(expects)
            return sum((e - o) ** 2 for e, o in zip(expects, outputs)) / n

        @staticmethod
        def mse_derivative(expect: float, output: float) -> float:
            return 2 * (output - expect)

        @staticmethod
        def binary_cross_entropy(expects, outputs):
            expects = Network._to_list(expects)
            outputs = Network._to_list(outputs)
            return -sum(
                (e * math.log(o)) + ((1 - e) * math.log(1 - o))
                for e, o in zip(expects, outputs)
            )

        @staticmethod
        def binary_cross_entropy_derivative(expect: float, output: float) -> float:
            return ((1 - expect) / (1 - output)) - (expect / output)

        @staticmethod
        def categorical_cross_entropy(expects, outputs):
            expects = Network._to_list(expects)
            outputs = Network._to_list(outputs)
            return -sum(e * math.log(o) for e, o in zip(expects, outputs))

    # ================================================================
    #  三、節點（Node）
    # ================================================================
    class _Node:
        """神經網路中的單一節點"""

        def __init__(
            self,
            pre_layer: "Network._Layer" = None,
            weights: Sequence[float] = None,
            node_type: str = "I",
            layer_idx: int = None,
            node_idx: int = None,
        ):
            self.id = self._generate_id(node_type, layer_idx, node_idx)
            self.pre_layer = pre_layer
            self.raw_output = None
            self.output = 1 if node_type == "B" else None  # 設定 bias output
            self.weight_mapping = {}
            self.d_TL_d_raw = None
            self.d_activation_output = 0
            self.gradient_mapping = {}

            # 若有前一層與權重，建立對應表
            weights = list(weights or [])
            if self.pre_layer and weights:
                if len(self.pre_layer.nodes) != len(weights):
                    raise ValueError(f"{self.id}: 前一層節點數與權重數量不一致")
                self.weight_mapping = dict(zip(self.pre_layer.nodes, weights))

        # ------------------------------------------------------------
        @staticmethod
        def _generate_id(node_type, layer_idx, node_idx):
            """自動生成節點 ID"""
            if node_type == "I":
                return f"I{node_idx + 1}"
            if node_type == "H":
                return f"H{(layer_idx or 0) + 1}_{node_idx + 1}"
            if node_type == "B":
                return f"B{(0 if layer_idx == None else layer_idx + 1)}"
            if node_type == "O":
                return f"O{node_idx + 1}"
            return f"N{node_idx}"

    # ================================================================
    #  四、層（Layer）
    # ================================================================
    class _Layer:
        """神經網路中的一層（input, hidden, output）"""

        def __init__(
            self, nodes: List["Network._Node"], layer_type: str, activation: str
        ):
            self.nodes = nodes
            self.type = layer_type
            self.activation = activation.lower()

        def __getitem__(self, idx):
            """支援索引與切片"""
            return self.nodes[idx]

        def __iter__(self):
            """支援迭代"""
            return iter(self.nodes)

    # ================================================================
    #  五、網路結構（NetworkMap）
    # ================================================================
    class NetworkMap:
        """儲存整體網路架構與顯示"""

        def __init__(self):
            self.inputs: Network._Layer = None
            self.hidden: List[Network._Layer] = []
            self.outputs: Network._Layer = None

        def show(self):
            """以文字方式顯示網路結構、連線與 activation"""
            print("\n=== Network Structure ===")

            # 輸入層
            print("\n[Input Layer] (activation: {})".format(self.inputs.activation))
            for n in self.inputs:
                print(f"  {n.id}")

            # 隱藏層
            for idx, layer in enumerate(self.hidden, start=1):
                print(f"\n[Hidden Layer {idx}] (activation: {layer.activation})")
                for node in layer.nodes:
                    if node.id.startswith("B"):
                        continue
                    for pre_node, w in node.weight_mapping.items():
                        print(f"  {pre_node.id:<6} --{w:>6.2f}--> {node.id}")

            # 輸出層
            print(f"\n[Output Layer] (activation: {self.outputs.activation})")
            for node in self.outputs.nodes:
                for pre_node, w in node.weight_mapping.items():
                    print(f"  {pre_node.id:<6} --{w:>6.2f}--> {node.id}")

            print("\n===========================\n")

        @property
        def layers(self):
            return [self.inputs] + self.hidden + [self.outputs]

        def __getitem__(self, idx):
            """支援索引與切片"""
            return self.layers[idx]

        def __iter__(self):
            """支援迭代"""
            return iter(self.layers)

    # ================================================================
    #  六、神經網路主體（Network）
    # ================================================================
    def __init__(self, network_setting_json: str):
        config: dict = json.loads(network_setting_json)
        self.map = Network.NetworkMap()

        # 建立 input nodes
        input_setting = config.get("input", {})
        self.map.inputs = self._build_layer(
            input_setting, pre_layer=None, layer_type="I"
        )

        pre_layer = self.map.inputs

        # 建立 hidden layers
        for layer_idx, layer_cfg in enumerate(config.get("layer", [])):
            layer = self._build_layer(
                layer_cfg, pre_layer, layer_type="H", layer_idx=layer_idx
            )
            self.map.hidden.append(layer)
            pre_layer = layer

        # 建立 output nodes
        output_cfg = config.get("output", {})
        self.map.outputs = self._build_layer(output_cfg, pre_layer, layer_type="O")

    # ------------------------------------------------------------
    def _build_layer(
        self,
        cfg: dict,
        pre_layer: "Network._Layer",
        layer_type: str,
        layer_idx: int = None,
    ) -> "Network._Layer":
        """建立一層（input / hidden / output）"""

        activation = cfg.get("activation", "linear").lower()
        node_count = cfg.get("nodes", 0)

        # input layer
        if layer_type == "I":
            nodes = [
                Network._Node(node_type="I", node_idx=i) for i in range(node_count)
            ]
            nodes.append(Network._Node(node_type="B"))  # bias node
            return Network._Layer(nodes, layer_type="I", activation=activation)

        # hidden/output layer
        weights_matrix = cfg.get("weights", [])
        bias_weights = cfg.get("bias_weights", [])
        nodes = []

        for node_idx in range(node_count):
            weights = list(weights_matrix[node_idx]) + [bias_weights[node_idx]]
            node = Network._Node(
                pre_layer=pre_layer,
                weights=weights,
                node_type=layer_type,
                layer_idx=layer_idx,
                node_idx=node_idx,
            )
            nodes.append(node)

        # Hidden 層需要 bias node
        if layer_type != "O":
            nodes.append(
                Network._Node(pre_layer=pre_layer, node_type="B", layer_idx=layer_idx)
            )
        return Network._Layer(nodes, layer_type=layer_type, activation=activation)

    # ------------------------------------------------------------
    def forward(self, input_values: Sequence[float]) -> List[float]:
        """執行前向傳遞（Forward propagation）"""

        input_values = Network._to_list(input_values)
        if len(input_values) != len(self.map.inputs.nodes) - 1:  # 忽略 bias
            raise ValueError("輸入數量與 input node 數量不一致")

        for val, node in zip(input_values, self.map.inputs[:-1]):
            node.output = val

        for layer, pre_layer in zip(self.map[1:], self.map[:-1]):
            # raw_output 計算
            nodes_raw_output_list = []
            for node in layer:
                if node.id.startswith("B"):
                    node.raw_output = 1
                    continue
                raw_output = sum(
                    pre_node.output * node.weight_mapping[pre_node]
                    for pre_node in pre_layer
                )
                node.raw_output = raw_output
                nodes_raw_output_list.append(raw_output)

            # activated_output 計算
            activation_func = getattr(
                Network.Activation,
                layer.activation,
                Network.Activation.linear,
            )
            activated_outputs = activation_func(nodes_raw_output_list)
            if len(layer.nodes) != len(activated_outputs):
                activated_outputs.append(1.0)  # bias output
            for node, output in zip(layer.nodes, activated_outputs):
                node.output = output

        return [node.output for node in self.map.outputs]

    # ------------------------------------------------------------
    def set_output_gradients(
        self,
        output_values: Sequence[float],
        expect_values: Sequence[float],
        loss_func: str,
    ):
        """手動設定outputs d_TL_d_raw"""
        output_values = Network._to_list(output_values)
        expect_values = Network._to_list(expect_values)
        if len(output_values) != len(expect_values):
            raise ValueError("outputs 數量與 expects 數量不一致")
        loss_func = loss_func.strip().lower()
        d_loss_func = getattr(Network.Loss, loss_func + "_derivative", None)
        output_layer_activation = self.map.outputs.activation
        d_output_layer_activation_func = getattr(
            Network.Activation,
            output_layer_activation + "_derivative",
            None,
        )
        for o, e, output_node in zip(output_values, expect_values, self.map.outputs):
            act_input = (
                output_node.output
                if output_layer_activation == "sigmoid"
                else output_node.raw_output
            )

            d_TL_d_raw = (
                d_loss_func(e, o) / (len(output_values) if loss_func == "mse" else 1)
            ) * d_output_layer_activation_func(act_input)
            output_node.d_TL_d_raw = d_TL_d_raw

            output_node.gradient_mapping = {
                pre_node: d_TL_d_raw * pre_node.output
                for pre_node in self.map.layers[-2]
            }

    # ------------------------------------------------------------
    def backward(self):
        """計算梯度"""

        rev_list = self.map.layers[::-1]
        for next_layer, layer, pre_layer in zip(rev_list, rev_list[1:], rev_list[2:]):

            d_activation = getattr(
                Network.Activation,
                layer.activation + "_derivative",
                Network.Activation.linear_derivative,
            )
            node_d_act_input_type = (
                "output" if layer.activation == "sigmoid" else "raw_output"
            )
            for node in layer:
                if node.id.startswith("B"):
                    continue
                d_activation_output = d_activation(getattr(node, node_d_act_input_type))
                node.d_activation_output = d_activation_output
                d_TL_d_raw = d_activation_output * (
                    sum(
                        next_layer_node.d_TL_d_raw
                        * next_layer_node.weight_mapping[node]
                        for next_layer_node in next_layer
                        if not next_layer_node.id.startswith("B")
                    )
                )
                node.d_TL_d_raw = d_TL_d_raw
                node.gradient_mapping = {
                    pre_node: d_TL_d_raw * pre_node.output for pre_node in pre_layer
                }

    # ------------------------------------------------------------
    def show_gradient(self) -> str:
        parts = ["====NetWork Current Gradient===="]
        for li, layer in enumerate(self.map[1:]):  # skip input layer
            # 判斷層名稱
            if layer.type == "O":
                layer_name = "Output Layer"
            else:
                layer_name = f"Hidden Layer {li+1}"

            parts.append(f"{layer_name}:")
            for node in layer:
                # 前一層名稱
                pre_layer_name = (
                    "Input Layer"
                    if node.pre_layer.type == "I"
                    else f"Hidden Layer {self.map.hidden.index(node.pre_layer)+1}"
                )
                grad_dict = {
                    pre_node.id: grad
                    for pre_node, grad in node.gradient_mapping.items()
                }
                parts.append(f"{node.id} <---> {pre_layer_name} {grad_dict}")
        return "\n".join(parts)

    # ------------------------------------------------------------
    def zero_grad(self, learning_rate: float):
        """更新權重並歸零梯度，回傳更新後的梯度字串"""
        for layer in self.map[1:]:
            for node in layer:
                for pre_node in node.weight_mapping:
                    node.weight_mapping[pre_node] -= (
                        node.gradient_mapping[pre_node] * learning_rate
                    )
                    node.gradient_mapping[pre_node] = 0

    # ------------------------------------------------------------
    def show_weights(self):
        """建立文字輸出"""
        parts = ["====NetWork Current weight===="]
        for li, layer in enumerate(self.map[1:]):
            if li == 0:
                layer_name = "Input Layer"
            elif li <= len(self.map.hidden):
                layer_name = f"Hidden Layer {li}"
            else:
                layer_name = "Output Layer"

            parts.append(f"{layer_name}:")

            for node in layer:
                if node.id.startswith("B"):
                    continue
                weight_dict = {
                    pre_node.id: w for pre_node, w in node.weight_mapping.items()
                }
                parts.append(f"  ---> {node.id} {weight_dict}")

        print("\n".join(parts))


    def generate_random_weight(self, start_num, end_num):
        for layer in self.map.layers[-1:0:-1]:
            for node in layer:
                if node.id.startswith("B"):
                    continue
                for key in node.weight_mapping:
                    node.weight_mapping[key] = random.uniform(start_num, end_num)

    @staticmethod
    def _to_list(x):
        if isinstance(x, tuple):
            return list(x)
        return x if isinstance(x, list) else [x]


# ===============================
# 執行
# ===============================

# ==========================================
# 讀取與輔助函數
# ==========================================
file_path = "gender-height-weight.csv" # 修正檔名 typo: ender -> gender

def read_csv_to_dict(path: str) -> List[Dict[str, str]]:
    data = []
    try:
        with open(path, mode="r", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(dict(row))
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {path}")
        return []
    return data

def calculate_stats(values):
    if not values: return 0, 1 # 防止除以零
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)
    return mean, std

# ==========================================
# Task 1
# ==========================================

def task1():
    # --- A. 讀取資料並轉換型別 ---
    raw_data_str = read_csv_to_dict(file_path)
    processed_data = []

    for row in raw_data_str:
        try:
            # 將字串轉為浮點數，並處理 Gender
            item = {
                'Gender': 0 if row['Gender'] == 'Male' else 1, # Male=0, Female=1
                'Height': float(row['Height']),
                'Weight': float(row['Weight'])
            }
            processed_data.append(item)
        except ValueError:
            continue # 跳過資料缺漏或格式錯誤的行

    # --- B. 切分訓練集與檢查集 (8:2) ---
    random.seed(42) # 固定種子，保證每次切分結果一樣
    random.shuffle(processed_data)

    split_idx = int(len(processed_data) * 0.8)
    train_set = processed_data[:split_idx]
    test_set = processed_data[split_idx:]

    print(f"資料載入完成: 總筆數 {len(processed_data)}")
    print(f"訓練集: {len(train_set)}, 檢查集: {len(test_set)}")

    # --- C. 計算統計數據 (只使用訓練集!) ---
    # 避免 Data Leakage: 測試集的資料不該影響標準化參數
    train_heights = [d['Height'] for d in train_set]
    train_weights = [d['Weight'] for d in train_set]

    h_mean, h_std = calculate_stats(train_heights)
    w_mean, w_std = calculate_stats(train_weights)

    print(f"統計參數 (Train): 身高均值={h_mean:.2f}, 體重均值={w_mean:.2f}")

    # --- D. 準備網路輸入資料 (標準化) ---
    def create_dataset(source_data):
        inputs = []
        expects = []
        for row in source_data:
            # Input: [Gender, Standardized_Height]
            h_norm = (row['Height'] - h_mean) / h_std
            inputs.append([row['Gender'], h_norm])
            
            # Expect: [Standardized_Weight]
            w_norm = (row['Weight'] - w_mean) / w_std
            expects.append([w_norm])
        return inputs, expects

    train_inputs, train_expects = create_dataset(train_set)
    test_inputs, test_expects = create_dataset(test_set)

    # --- E. 初始化網路 ---
    model_2h_json = """
    {
    "input": {
        "nodes": 2,
        "activation": "linear"
    },
    "layer": [
        {
        "nodes": 2,
        "activation": "sigmoid",
        "weights": [[0.1, -0.1], [-0.1, 0.1]],
        "bias_weights": [0.1, -0.1]
        }
    ],
    "output": {
        "nodes": 1,
        "activation": "linear",
        "weights": [[0.1, -0.1]],
        "bias_weights": [0.1]
    }
    }
    """

    # 假設這是你的 Network 初始化方式
    net = Network(model_2h_json)

    # 注意：你原本 JSON 裡有寫死 weights，但這裡呼叫 random_weight 會把 JSON 的覆蓋掉
    # 這通常是正確的，因為訓練前我們希望隨機初始化
    net.generate_random_weight(-0.1, 0.1)
    print("初始權重:")
    net.show_weights()

    # 設定 Loss Function (回歸問題使用 MSE)
    loss_func = Network.Loss.mse
    loss_derivative = Network.Loss.mse_derivative

    # --- F. 訓練迴圈 (Training Loop) ---
    learning_rate = 0.05
    epochs = 100  # 訓練幾輪

    print("\n開始訓練...")
    for epoch in range(epochs):
        
        if epoch == 40:
            learning_rate = learning_rate / 10
            print(f"\n[訊息] Epoch {epoch}: 學習率已調整為 {learning_rate}\n")
            
        if epoch == 70:
            learning_rate = learning_rate / 10
            print(f"\n[訊息] Epoch {epoch}: 學習率已調整為 {learning_rate}\n")
        
        total_loss = 0
        
        for x, y in zip(train_inputs, train_expects):
            # 1. Forward
            output = net.forward(x)
            
            # 2. Calculate Loss (僅供觀察)
            loss = loss_func(y, output)
            total_loss += loss
            
            # 3. Backward (這部分依賴你 Network 類別的實作，假設有這個方法)
            net.set_output_gradients(output, y, "mse")
            net.backward() 
            net.zero_grad(learning_rate)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg Loss = {total_loss / len(train_inputs):.6f}")

    # --- G. 最終測試評估 (Evaluation) ---
    print("\n========================================")
    print("開始最終測試 (評估整個檢查集)...")
    print("========================================")

    total_test_loss = 0
    total_abs_error_lbs = 0  # 用來累積絕對誤差 (磅)

    # 走訪每一筆測試資料
    for x, y in zip(test_inputs, test_expects):
        
        # 1. Forward (只做前向傳播，不做反向傳播!)
        output = net.forward(x)
        
        # 2. 取得預測值與真實值 (Z-score)
        pred_z = output[0]
        real_z = y[0]
        
        # 3. 計算 MSE (標準化空間的 Loss) - 用來跟訓練時的 Loss 比較
        loss = (pred_z - real_z) ** 2
        total_test_loss += loss
        
        # 4. 還原成真實單位 (磅)
        pred_lbs = (pred_z * w_std) + w_mean
        real_lbs = (real_z * w_std) + w_mean
        
        # 5. 計算絕對誤差 (MAE) - 這是給人類看的指標
        # 我們想知道平均預測差幾磅，而不是差幾磅的平方
        total_abs_error_lbs += abs(pred_lbs - real_lbs)

    # --- 計算平均值 ---
    avg_test_mse = total_test_loss / len(test_inputs)
    avg_mae_lbs = total_abs_error_lbs / len(test_inputs)

    print(f"測試集樣本數: {len(test_inputs)}")
    print(f"最終 MSE Loss (Z-score): {avg_test_mse:.6f}")
    print(f"平均絕對誤差 (MAE): {avg_mae_lbs:.2f} lbs")

    # --- (選用) 顯示前 5 筆預測結果給你看感覺 ---
    print("\n[前 5 筆預測抽樣]")
    for i in range(5):
        out = net.forward(test_inputs[i])[0]
        p_lbs = (out * w_std) + w_mean
        r_lbs = (test_expects[i][0] * w_std) + w_mean
        print(f"樣本 {i+1}: 預測 {p_lbs:.1f} lbs | 實際 {r_lbs:.1f} lbs | 誤差 {abs(p_lbs - r_lbs):.1f} lbs")
        
    print("最後權重:")
    net.show_weights()

# ==========================================
# Task 2
# ==========================================

def generate_minimal_one_hot_encoding(
    row_data: Dict[str, str], 
    column_name: str, 
    regex_pattern: str
) -> Dict[str, int]:
    """
    對單行數據的指定欄位進行獨熱編碼。
    只為符合正規表達式、且成功提取的片段創建 Key (Value 為 1)。
    
    Args:
        row_data: 單行數據的字典表示 (key為欄位名，value為值)。
        column_name: 要編碼的欄位名稱 (例如 'Name')。
        regex_pattern: 用於提取片段的正規表達式。
        
    Returns:
        一個包含獨熱編碼結果的字典 (例如 {'Name_Mr': 1})。
    """
    one_hot_item = {}
    
    if column_name not in row_data:
        return one_hot_item
    
    cell_value = row_data[column_name]
    
    # 提取當前行的匹配片段
    match = re.search(regex_pattern, cell_value)
    
    if match:
        # 提取第一個捕獲組（即稱謂片段）
        extracted_segment = match.group(1) 
        
        # 創建 Key：'欄位名_片段'
        new_key = f"{column_name}_{extracted_segment}"
        
        # 設置 Value 為 1
        one_hot_item[new_key] = 1
            
    return one_hot_item


WHOLE_STRING_REGEX = r'^(.+)$'
TITLE_REGEX = r' ([A-Za-z]+)\.'

raw_data = read_csv_to_dict("titanic.csv")
processed_data = []

for row in raw_data:
    try:

        item = {}
        # 處理Pclass
        pclass_dict = generate_minimal_one_hot_encoding(row, "Pclass" , WHOLE_STRING_REGEX)
        item.update(pclass_dict)

        # 處理Name
        name_dict = generate_minimal_one_hot_encoding(row, "Name", TITLE_REGEX)
        item.update(name_dict)
        
        # 處理Sex
        item["Sex"] = 1 if row["Sex"] == "male" else 0

        # 處理Age
        item["Age"] = row["Age"]        

        # 處理SibSp
        item["SibSp"] = row["SibSp"]        

        # 處理Parch
        item["Parch"] = row["Parch"]        

       

        processed_data.append(item)
    except ValueError:
        continue # 跳過資料缺漏或格式錯誤的行

