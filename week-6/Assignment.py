from collections import Counter
import json
import math
import csv
import random
import re
from typing import Any, Sequence, List, Dict, Tuple


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

    @staticmethod
    def build_model_json(
        input_nodes,
        hidden_nodes_list,
        output_nodes,
        hidden_activation="sigmoid",
        output_activation="linear",
        input_activation="linear",
    ):

        def rand():
            return random.uniform(-0.1, 0.1)

        def weight_matrix(rows, cols):
            return [[rand() for _ in range(cols)] for _ in range(rows)]

        def bias_vector(size):
            return [rand() for _ in range(size)]

        model = {}

        # input
        model["input"] = {"nodes": input_nodes, "activation": input_activation}

        # hidden layers
        layers = []
        prev_nodes = input_nodes
        for h_nodes in hidden_nodes_list:
            layer = {
                "nodes": h_nodes,
                "activation": hidden_activation,
                "weights": weight_matrix(h_nodes, prev_nodes),
                "bias_weights": bias_vector(h_nodes),
            }
            layers.append(layer)
            prev_nodes = h_nodes

        model["layer"] = layers

        # output layer
        model["output"] = {
            "nodes": output_nodes,
            "activation": output_activation,
            "weights": weight_matrix(output_nodes, prev_nodes),
            "bias_weights": bias_vector(output_nodes),
        }

        return model


# =====================================================
# 執行
# =====================================================
# ==========================================
# I. 全域共用設定和輔助函數
# ==========================================
TEST_SIZE_RATIO = 0.2
RANDOM_SEED = 42


# --- CSV 讀取 ---
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


# --- 統計計算 ---
def calculate_mean(data_list: list):
    clean_list = [x for x in data_list if x is not None]
    return sum(clean_list) / len(clean_list) if clean_list else 0.0


def calculate_std(data_list: list, mean: float):
    clean_list = [x for x in data_list if x is not None]
    if len(clean_list) <= 1:
        return 1.0
    variance = sum([(x - mean) ** 2 for x in clean_list]) / (len(clean_list) - 1)
    return math.sqrt(variance)


def calculate_median(data_list: list):
    clean_list = [x for x in data_list if x is not None]
    if not clean_list:
        return 0.0
    clean_list.sort()
    N = len(clean_list)
    return (
        (clean_list[N // 2 - 1] + clean_list[N // 2]) / 2
        if N % 2 == 0
        else clean_list[N // 2]
    )


def calculate_stats(values):
    # Task 1 中使用的 calculate_stats (用於計算 Mean 和 Std)
    if not values:
        return 0, 1
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)
    return mean, std


# --- 數據拆分 ---
def train_test_split_pure_python(
    features: List[List[float]], labels: List[Any], test_ratio: float, seed: int = 42
) -> Tuple[List[List[float]], List[List[float]], List[Any], List[Any]]:

    random.seed(seed)
    data_size = len(features)

    if data_size != len(labels):
        raise ValueError("特徵矩陣和標籤的行數必須一致。")

    test_size = int(data_size * test_ratio)
    indices = list(range(data_size))
    random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # 這裡的 labels 是 Any 類型，因為 Task 1 的標籤是 [float]，Task 2 是 int
    X_train = [features[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    Y_train = [labels[i] for i in train_indices]
    Y_test = [labels[i] for i in test_indices]

    return X_train, X_test, Y_train, Y_test


# ==========================================
# II. Task 1: Gender-Height-Weight (迴歸)
# ==========================================

GENDER_FILE = "gender-height-weight.csv"


def preprocess_and_standardize_gender_data(
    raw_data_str: List[Dict[str, str]],
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], float, float, float, float]:
    """
    處理 Task 1 的數據，但不在這裡拆分和轉換為最終矩陣，而是返回處理後的字典列表和統計參數。
    """
    processed_data = []

    for row in raw_data_str:
        try:
            # A. 讀取資料並轉換型別 (Gender: Male=0, Female=1)
            item = {
                "Gender": 0 if row["Gender"] == "Male" else 1,
                "Height": float(row["Height"]),
                "Weight": float(row["Weight"]),
            }
            processed_data.append(item)
        except ValueError:
            continue

    # B. 切分訓練集與檢查集
    random.seed(RANDOM_SEED)
    random.shuffle(processed_data)
    split_idx = int(len(processed_data) * 0.8)
    train_set = processed_data[:split_idx]
    test_set = processed_data[split_idx:]

    # C. 計算統計數據 (只使用訓練集)
    train_heights = [d["Height"] for d in train_set]
    train_weights = [d["Weight"] for d in train_set]
    h_mean, h_std = calculate_stats(train_heights)
    w_mean, w_std = calculate_stats(train_weights)

    return train_set, test_set, h_mean, h_std, w_mean, w_std


# ==========================================
# III. Task 2: Titanic (分類)
# ==========================================

TITANIC_FILE = "titanic.csv"
REGEX = {
    "WHOLE_STRING": r"^(.+)$",
    "NAME_TITLE": r" ([A-Za-z]+)\.",
    "FIRST_LETTER": r"([A-Z])",
}


# --- 獨熱編碼  ---
def generate_minimal_one_hot_encoding(
    row_data: Dict[str, Any], column_name: str, regex_pattern: str, create_None: bool = False
) -> Dict[str, int]:
    one_hot_item = {}
    if column_name not in row_data:
        return one_hot_item
    cell_value = row_data[column_name]
    cell_value = str(cell_value) if cell_value is not None else ""
    match = re.search(regex_pattern, cell_value)

    if match:
        extracted_segment = match.group(1)
        one_hot_item[f"{column_name}_{extracted_segment}"] = 1
    elif create_None:
        one_hot_item[f"{column_name}_None"] = 1
    return one_hot_item


def preprocess_and_standardize_titanic_data(
    raw_data: List[Dict[str, Any]],
) -> Tuple[List[List[float]], List[int], int]:

    # --- Stage 1A: 分離目標標籤 (Y) 和初始化特徵列表 (X) ---
    target_labels = []
    data_for_processing = []

    for row in raw_data:
        try:
            survived_value = row.pop("Survived")
            target_labels.append(int(survived_value))
            data_for_processing.append(row)
        except (KeyError, ValueError):
            data_for_processing.append(row)
            continue

    # =================================================================
    # 🔥 新增步驟：自動統計稱謂頻率 (讓數據決定誰是稀有)
    # =================================================================
    all_raw_titles = []
    for row in data_for_processing:
        name_str = str(row.get("Name", ""))
        match = re.search(REGEX["NAME_TITLE"], name_str)
        if match:
            all_raw_titles.append(match.group(1))
    
    # 計算每個稱謂出現的次數
    title_counts = Counter(all_raw_titles)
    
    # 設定門檻：出現少於 10 次的都當作稀有
    MIN_FREQUENCY = 10 
    
    # 建立「常見稱謂」的白名單 (Set 查詢速度快)
    # 例如：{'Mr', 'Miss', 'Mrs', 'Master'}
    common_titles = {title for title, count in title_counts.items() if count >= MIN_FREQUENCY}
    
    print(f"[系統自動偵測] 常見稱謂 (出現次數>={MIN_FREQUENCY}): {common_titles}")
    # ================================================================= 

    processed_data = []
    all_keys = {
        "Pclass": set(),
        "Name": set(),
        "Cabin": set(),
        "Embarked": set(),
    }

    # --- Stage 1B: 特徵提取與缺失標註 (X) ---
    for row in data_for_processing:
        item = {}

        # 處理類別特徵
        pclass_dict = generate_minimal_one_hot_encoding(
            row, "Pclass", REGEX["WHOLE_STRING"]
        )
        item.update(pclass_dict)
        all_keys["Pclass"].update(pclass_dict.keys())
        
        # -------------------------------------------------------------
        # 🔥 修改 Name 處理邏輯：動態映射
        # -------------------------------------------------------------
        name_str = str(row.get("Name", ""))
        match = re.search(REGEX["NAME_TITLE"], name_str)
        
        clean_title = "Rare" # 預設為稀有
        if match:
            raw_title = match.group(1)
            # 檢查這個稱謂是否在我們剛算出來的白名單內
            if raw_title in common_titles:
                clean_title = raw_title
            else:
                clean_title = "Rare"
        
        # 手動建立 One-Hot 鍵值
        title_key = f"Name_{clean_title}"
        item[title_key] = 1
        all_keys["Name"].add(title_key)
        
        
        item["Sex"] = 1 if row["Sex"] == "male" else 0

        # 處理Age (缺失標註)
        age_value = row.get("Age")
        is_age_missing = age_value is None or age_value == ""
        item["Age_Missing"] = 1 if is_age_missing else 0
        item["Age"] = (
            float(age_value) if not is_age_missing and age_value is not None else None
        )

        # 處理SibSp 和 Parch (轉為 float)
        item["SibSp"] = float(row["SibSp"])
        item["Parch"] = float(row["Parch"])

        # 處理Fare (缺失標註)
        fare_value = row.get("Fare")
        is_fare_missing = fare_value is None or fare_value == ""
        item["Fare_Missing"] = 1 if is_fare_missing else 0
        item["Fare"] = (
            float(fare_value)
            if not is_fare_missing and fare_value is not None
            else None
        )

        # 處理Cabin 和 Embarked
        cabin_dict = generate_minimal_one_hot_encoding(
            row, "Cabin", REGEX["FIRST_LETTER"], True
        )
        item.update(cabin_dict)
        all_keys["Cabin"].update(cabin_dict.keys())
        embarked_dict = generate_minimal_one_hot_encoding(
            row, "Embarked", REGEX["WHOLE_STRING"], True
        )
        item.update(embarked_dict)
        all_keys["Embarked"].update(embarked_dict.keys())

        processed_data.append(item)

    # --- Stage 2A/2B: 填補、對數轉換 (Fare) 並收集數據 ---
    all_ages = [row.get("Age") for row in processed_data]
    all_fares = [row.get("Fare") for row in processed_data]
    MEDIAN_AGE = calculate_median(all_ages)
    MEDIAN_FARE = calculate_median(all_fares)
    temp_age_list, temp_fare_list, temp_sibsp_list, temp_parch_list = [], [], [], []

    for row in processed_data:
        if row["Age"] is None:
            row["Age"] = MEDIAN_AGE
        temp_age_list.append(row["Age"])

        if row["Fare"] is None:
            row["Fare"] = MEDIAN_FARE
        row["Fare"] = math.log(row["Fare"] + 1)
        temp_fare_list.append(row["Fare"])

        temp_sibsp_list.append(row["SibSp"])
        temp_parch_list.append(row["Parch"])

    # --- Stage 2C/2D: 計算和執行 Z-Score 標準化 ---
    MEAN_AGE = calculate_mean(temp_age_list)
    STD_AGE = calculate_std(temp_age_list, MEAN_AGE)
    MEAN_FARE = calculate_mean(temp_fare_list)
    STD_FARE = calculate_std(temp_fare_list, MEAN_FARE)
    MEAN_SIBSP = calculate_mean(temp_sibsp_list)
    STD_SIBSP = calculate_std(temp_sibsp_list, MEAN_SIBSP)
    MEAN_PARCH = calculate_mean(temp_parch_list)
    STD_PARCH = calculate_std(temp_parch_list, MEAN_PARCH)

    for row in processed_data:
        row["Age"] = (row["Age"] - MEAN_AGE) / STD_AGE
        row["Fare"] = (row["Fare"] - MEAN_FARE) / STD_FARE
        row["SibSp"] = (row["SibSp"] - MEAN_SIBSP) / STD_SIBSP
        row["Parch"] = (row["Parch"] - MEAN_PARCH) / STD_PARCH

    # --- Stage 2E: 統一特徵空間並轉換為矩陣 (final_feature_matrix) ---
    SCALAR_KEYS = [
        "Sex",
        "Age",
        "Age_Missing",
        "SibSp",
        "Parch",
        "Fare",
        "Fare_Missing",
    ]
    ONE_HOT_KEYS = [
        k
        for key_type in ["Pclass", "Name", "Cabin", "Embarked"]
        for k in sorted(list(all_keys[key_type]))
    ]
    FINAL_FEATURE_MAP = SCALAR_KEYS + ONE_HOT_KEYS

    print("FINAL_FEATURE_MAP = " ,FINAL_FEATURE_MAP)

    final_feature_matrix = []
    for row_dict in processed_data:
        feature_vector = [float(row_dict.get(key, 0.0)) for key in FINAL_FEATURE_MAP]
        final_feature_matrix.append(feature_vector)

    return final_feature_matrix, target_labels, len(FINAL_FEATURE_MAP)

def build_titanic_model_config(input_nodes: int):
    """
    [Task 2 輔助] 生成 Titanic 神經網路模型的 JSON 設定結構
    """
    # 隱藏層結構：嘗試兩層，節點數介於輸入(約12-15)與輸出(1)之間
    hidden_nodes_list = [20, 10, 4] 
    
    # 輸出層：1 個節點 (生存機率)，激活函數一定要用 Sigmoid (將輸出壓在 0~1)
    output_nodes = 1
    output_activation = "sigmoid" 
    hidden_activation = "sigmoid" 

    # 呼叫 Network 類別的靜態方法
    model_config = Network.build_model_json(
        input_nodes=input_nodes,
        hidden_nodes_list=hidden_nodes_list,
        output_nodes=output_nodes,
        hidden_activation=hidden_activation,
        output_activation=output_activation
    )
    return model_config


def calculate_accuracy(predictions: List[List[float]], targets: List[int]) -> float:
    """
    [Task 2 輔助] 計算分類準確率 (閾值 0.5)
    """
    correct = 0
    total = len(targets)
    if total == 0: return 0.0
    
    for pred_vector, target in zip(predictions, targets):
        # pred_vector[0] 是模型輸出的機率 (例如 0.72)
        # 如果 >= 0.5 預測為 1 (存活)，否則為 0 (死亡)
        pred_label = 1 if pred_vector[0] >= 0.5 else 0
        
        if pred_label == target:
            correct += 1
            
    return correct / total

# ==========================================
# IV. 主執行流程
# ==========================================


def run_task1_training():
    """執行 Gender-Height-Weight 迴歸訓練和評估"""

    raw_data_str = read_csv_to_dict(GENDER_FILE)

    # 執行數據處理，得到訓練/測試集和統計參數
    train_set, test_set, h_mean, h_std, w_mean, w_std = (
        preprocess_and_standardize_gender_data(raw_data_str)
    )

    print(f"資料載入完成: 總筆數 {len(train_set) + len(test_set)}")
    print(f"訓練集: {len(train_set)}, 檢查集: {len(test_set)}")
    print(f"統計參數 (Train): 身高均值={h_mean:.2f}, 體重均值={w_mean:.2f}")

    # 準備網路輸入資料 (標準化)
    def create_dataset(source_data):
        inputs = []
        expects = []
        for row in source_data:
            h_norm = (row["Height"] - h_mean) / h_std
            inputs.append([row["Gender"], h_norm])
            w_norm = (row["Weight"] - w_mean) / w_std
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
        print(
            f"樣本 {i+1}: 預測 {p_lbs:.1f} lbs | 實際 {r_lbs:.1f} lbs | 誤差 {abs(p_lbs - r_lbs):.1f} lbs"
        )

    print("最後權重:")
    net.show_weights()


def run_task2_training():
    """執行 Titanic 分類訓練、配置與評估"""
    
    print("\n[Step 1] 讀取與預處理數據...")
    raw_data = read_csv_to_dict(TITANIC_FILE)
    
    # 執行特徵工程
    final_feature_matrix, target_labels, input_nodes = (
        preprocess_and_standardize_titanic_data(raw_data)
    )
    
    # 執行數據拆分
    X_train, X_test, Y_train, Y_test = train_test_split_pure_python(
        final_feature_matrix, target_labels, TEST_SIZE_RATIO, RANDOM_SEED
    )

    print(f"特徵數: {input_nodes}")
    print(f"訓練集: {len(X_train)} 筆, 測試集: {len(X_test)} 筆")

    # [Step 2] 建立模型配置並初始化
    print("\n[Step 2] 建立神經網路模型...")
    model_config = build_titanic_model_config(input_nodes)
    
    # 初始化網路 (將 dict 轉為 json string 傳入)
    net = Network(json.dumps(model_config)) 
    net.generate_random_weight(-0.1, 0.1) # 隨機初始化權重

    # [Step 3] 設定訓練參數
    # 如果你的 Network 支援 "cross_entropy"，這裡改用它會更好；否則使用 "mse"
    loss_name = "binary_cross_entropy" 
    loss_func = Network.Loss.binary_cross_entropy 
    
    learning_rate = 0.05 
    epochs = 200        # 需要較多輪次來收斂

    print(f"\n[Step 3] 開始訓練 (Epochs: {epochs}, LR: {learning_rate})...")
    
    for epoch in range(epochs):
        total_loss = 0
        train_preds = []

        # --- 批次訓練 (Stochastic Gradient Descent) ---
        for x, y_scalar in zip(X_train, Y_train):
            # y_scalar 是 int (0 或 1)，神經網路通常期待列表格式 [0] 或 [1]
            y_target = [y_scalar]

            # 1. Forward
            output = net.forward(x)
            train_preds.append(output)

            # 2. Backward
            net.set_output_gradients(output, y_target, loss_name) 
            net.backward()
            net.zero_grad(learning_rate)
            
            # 累積 Loss (僅供觀察)
            total_loss += loss_func(y_target, output)

        # 每 20 輪印出一次狀態
        if epoch % 20 == 0:
            avg_loss = total_loss / len(X_train)
            acc = calculate_accuracy(train_preds, Y_train)
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f} | Train Acc = {acc*100:.1f}%")

    print("\n最終權重")
    net.show_weights()
    # [Step 4] 最終測試評估
    print("\n[Step 4] 最終測試集評估...")
    
    test_preds = []
    for x in X_test:
        output = net.forward(x)
        test_preds.append(output)
    
    final_acc = calculate_accuracy(test_preds, Y_test)
    print(f"========================================")
    print(f"🏆 Titanic 測試集準確率: {final_acc*100:.2f}%")
    print(f"========================================")
    
    # 顯示前 5 筆預測詳情
    print("\n[前 5 筆測試樣本預測詳情]")
    for i in range(min(5, len(X_test))):
        prob = test_preds[i][0]
        pred = "生還" if prob >= 0.5 else "死亡"
        actual = "生還" if Y_test[i] == 1 else "死亡"
        status = "✅" if pred == actual else "❌"
        print(f"樣本 {i}: 預測機率 {prob:.4f} ({pred}) | 實際: {actual} {status}")


if __name__ == "__main__":

    print("========================================")
    print("        🚀 執行 Task 1: Gender-Weight (迴歸) 🚀")
    print("========================================")
    run_task1_training() # 執行迴歸任務的完整流程

    print("\n========================================")
    print("        🚢 執行 Task 2: Titanic (分類) 🚢")
    print("========================================")
    # run_task2_training()
