import json
import math
from typing import Sequence, List
from datetime import datetime


class Network:

    # ================================================================
    #  一、Activation Functions
    # ================================================================
    class Activation:
        def __init__(self):
            pass

        @staticmethod
        def linear(x: Sequence[float]) -> List[float]:
            return x

        @staticmethod
        def relu(x: Sequence[float]) -> List[float]:
            return [max(0, v) for v in x]

        @staticmethod
        def sigmoid(x: Sequence[float]) -> List[float]:
            return [1 / (1 + math.exp(-v)) for v in x]

        @staticmethod
        def softmax(x: Sequence[float]) -> List[float]:
            """Softmax：適用於分類輸出層"""
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
        def mse(y_true, y_pred):
            """Mean Squared Error"""
            y_true = Network.Loss._to_list(y_true)
            y_pred = Network.Loss._to_list(y_pred)
            n = len(y_true)
            return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

        @staticmethod
        def binary_cross_entropy(y_true, y_pred):
            """Binary Cross Entropy"""
            y_true = Network.Loss._to_list(y_true)
            y_pred = Network.Loss._to_list(y_pred)
            return -sum(
                (yt * math.log(yp)) + ((1 - yt) * math.log(1 - yp))
                for yt, yp in zip(y_true, y_pred)
            )

        @staticmethod
        def categorical_cross_entropy(y_true, y_pred):
            """Categorical Cross Entropy"""
            y_true = Network.Loss._to_list(y_true)
            y_pred = Network.Loss._to_list(y_pred)
            return -sum(yt * math.log(yp) for yt, yp in zip(y_true, y_pred))

        @staticmethod
        def _to_list(x):
            if isinstance(x, (int, float)):
                return [x]
            return list(x)

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
            self._output = 0.0  # activation 後的值
            self._raw_output = 0.0  # activation 前的值
            self.id = self._generate_id(node_type, layer_idx, node_idx)

            self.pre_layer = pre_layer
            self.weights = list(weights or [])
            self._weight_mapping = {}

            # 若有前一層與權重，建立對應表
            if self.pre_layer and self.weights:
                if len(self.pre_layer.nodes) != len(self.weights):
                    raise ValueError(f"{self.id}: 前一層節點數與權重數量不一致")
                self._weight_mapping = dict(zip(self.pre_layer.nodes, self.weights))

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

        # ------------------------------------------------------------
        @property
        def raw_output(self):
            """取得 activation 前的輸出（線性加權結果）"""
            if self.id.startswith("B"):  # Bias node
                return 1.0
            if self.id.startswith("I"):  # Input node
                return self._output
            return sum(
                self._weight_mapping[pre_node] * pre_node_output
                for pre_node, pre_node_output in zip(
                    self.pre_layer.nodes, self.pre_layer.outputs
                )
            )

        @property
        def output(self):
            """activation 後的最終輸出"""
            return self._output

        @output.setter
        def output(self, value):
            self._output = value

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

        @property
        def outputs(self) -> List[float]:
            """執行本層 activation 並回傳輸出結果"""

            ori_outputs = list(node.raw_output for node in self.nodes)
            activation_func = getattr(
                Network.Activation,
                self.activation,
                Network.Activation.linear,
            )
            activated_outputs = activation_func(ori_outputs)
            for node, val in zip(self.nodes, activated_outputs):
                node.output = val
            return activated_outputs

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
                    for pre_node, w in node._weight_mapping.items():
                        print(f"  {pre_node.id:<6} --{w:>6.2f}--> {node.id}")

            # 輸出層
            print(f"\n[Output Layer] (activation: {self.outputs.activation})")
            for node in self.outputs.nodes:
                for pre_node, w in node._weight_mapping.items():
                    print(f"  {pre_node.id:<6} --{w:>6.2f}--> {node.id}")

            print("\n===========================\n")

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

        # # 顯示網路架構
        # self.map.show()

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

        if len(input_values) != len(self.map.inputs.nodes) - 1:
            raise ValueError("輸入數量與 input node 數量不一致")

        for val, node in zip(input_values, self.map.inputs[:-1]):
            node.output = val

        return self.map.outputs.outputs


# ===============================
# Console 互動輸入模型設定並生成 JSON
# ===============================
def input_network_from_console():
    # === Input layer ===
    input_count = int(input("Input node 數量(不計bias): "))
    input_activation = (
        input(
            "Input layer activation (Linear / ReLU / Sigmoid / Softmax (大小寫不限)，預設 linear): "
        )
        .strip()
        .lower()
    )
    if not input_activation:
        input_activation = "linear"

    # === Hidden layers ===
    hidden_str = input(
        "Hidden layers 節點數量，不計bias (如總共三層，第一到三層內部的節點數分別為3、2、4則輸入3,2,4): "
    )
    hidden_layers = [int(x) for x in hidden_str.split(",") if x.strip()]

    # === Output layer ===
    output_count = int(input("Output node 數量: "))
    output_activation = (
        input(
            "Output layer activation (Linear / ReLU / Sigmoid / Softmax (大小寫不限)，預設 linear): "
        )
        .strip()
        .lower()
    )
    if not output_activation:
        output_activation = "linear"

    # === JSON 主體結構 ===
    network_dict = {
        "input": {"nodes": input_count, "activation": input_activation},
        "layer": [],
        "output": {
            "nodes": output_count,
            "weights": [],
            "bias_weights": [],
            "activation": output_activation,
        },
    }

    prev_nodes = input_count

    # === Hidden layers 權重設定 ===
    for layer_idx, nodes_count in enumerate(hidden_layers):
        print(f"\n=== Hidden Layer {layer_idx+1} 權重設定 ===")

        weights = []
        for node_idx in range(nodes_count):
            val = input(
                f"請輸入第 {node_idx+1} 個節點來自前一層的權重，用逗號分隔 (前一層有 {prev_nodes} 個節點): "
            )
            w_list = [float(x) for x in val.split(",")]
            if len(w_list) != prev_nodes:
                raise ValueError(f"權重數量必須是 {prev_nodes}")
            weights.append(w_list)

        # bias 權重
        bias_input = input(
            f"請輸入 Hidden Layer {layer_idx+1} 每個節點從前一層 bias 計算的權重，用逗號分隔(本層有{nodes_count}個節點): "
        )
        bias_weights = [float(x) for x in bias_input.split(",")]
        if len(bias_weights) != nodes_count:
            raise ValueError(f"bias 權重數量必須是 {nodes_count}")

        # activation
        activation = (
            input(
                f"請輸入 Hidden Layer {layer_idx+1} 的 activation (Linear / ReLU / Sigmoid / Softmax (大小寫不限)，預設 linear): "
            )
            .strip()
            .lower()
        )
        if not activation:
            activation = "linear"

        network_dict["layer"].append(
            {
                "nodes": nodes_count,
                "weights": weights,
                "bias_weights": bias_weights,
                "activation": activation,
            }
        )

        prev_nodes = nodes_count

    # === Output layer 權重設定 ===
    print("\n=== Output Layer 權重設定 ===")
    weights = []
    for node_idx in range(output_count):
        val = input(
            f"請輸入第 {node_idx+1} 個輸出節點來自前一層的權重，用逗號分隔 (前一層有 {prev_nodes} 個節點): "
        )
        w_list = [float(x) for x in val.split(",")]
        if len(w_list) != prev_nodes:
            raise ValueError(f"權重數量必須是 {prev_nodes}")
        weights.append(w_list)

    bias_input = input(
        f"請輸入 Output Layer 每個節點來自前一層 bias 的權重，用逗號分隔(本層有{output_count}個節點): "
    )
    bias_weights = [float(x) for x in bias_input.split(",")]
    if len(bias_weights) != output_count:
        raise ValueError(f"bias 權重數量必須是 {output_count}")

    network_dict["output"]["weights"] = weights
    network_dict["output"]["bias_weights"] = bias_weights

    # === 輸出 JSON 字串 ===
    return json.dumps(network_dict, indent=2)


# ===============================
# 執行
# ===============================
# 第一個模型 JSON 字串
model_1_json = """
{
  "input": {"nodes":2,
  "activation": "linear"},
  "layer": [
    {
      "nodes": 2,
      "weights": [
        [0.5, 0.2],
        [0.6, -0.6]
      ],
      "bias_weights": [0.3, 0.25],
      "activation": "relu"
    }
  ],
  "output": {
    "nodes": 2,
    "weights": [
      [0.8, -0.5],
      [0.4, 0.5]
    ],
    "bias_weights": [0.6, -0.25],
    "activation": "linear"
  }
}
"""

# 第二個模型 JSON 字串
model_2_json = """
{
  "input": {
    "nodes": 2,
    "activation": "linear"
  },
  "layer": [
    {
      "nodes": 2,
      "weights": [
        [
          0.5,
          0.2
        ],
        [
          0.6,
          -0.6
        ]
      ],
      "bias_weights": [
        0.3,
        0.25
      ],
      "activation": "relu"
    }
  ],
  "output": {
    "nodes": 1,
    "weights": [
      [
        0.8,
        0.4
      ]
    ],
    "bias_weights": [
      -0.5
    ],
    "activation": "sigmoid"
  }
}
"""
# 第三個模型 JSON 字串
model_3_json = """
{
  "input": {
    "nodes": 2,
    "activation": "linear"
  },
  "layer": [
    {
      "nodes": 2,
      "weights": [
        [
          0.5,
          0.2
        ],
        [
          0.6,
          -0.6
        ]
      ],
      "bias_weights": [
        0.3,
        0.25
      ],
      "activation": "relu"
    }
  ],
  "output": {
    "nodes": 3,
    "weights": [
      [
        0.8,
        -0.4
      ],
      [
        0.5,
        0.4
      ],
      [
        0.3,
        0.75
      ]
    ],
    "bias_weights": [
      0.6,
      0.5,
      -0.5
    ],
    "activation": "sigmoid"
  }
}
"""

# 第四個模型 JSON 字串
model_4_json = """
{
  "input": {
    "nodes": 2,
    "activation": "linear"
  },
  "layer": [
    {
      "nodes": 2,
      "weights": [
        [
          0.5,
          0.2
        ],
        [
          0.6,
          -0.6
        ]
      ],
      "bias_weights": [
        0.3,
        0.25
      ],
      "activation": "relu"
    }
  ],
  "output": {
    "nodes": 3,
    "weights": [
      [
        0.8,
        -0.4
      ],
      [
        0.5,
        0.4
      ],
      [
        0.3,
        0.75
      ]
    ],
    "bias_weights": [
      0.6,
      0.5,
      -0.5
    ],
    "activation": "softmax"
  }
}
"""
print("程式開始")
print("模型一：")
net = Network(model_1_json)
print("input = (1.5, 0.5)")
outputs = net.forward((1.5, 0.5))
print("outputs:", outputs)
print("expects = (0.8, 1)")
expects = (0.8, 1)
print("Total Loss:", Network.Loss.mse(expects, outputs))
print()
print("input = (0, 1)")
outputs = net.forward((0, 1))
print("outputs:", outputs)
print("expects = (0.5, 0.5)")
expects = (0.5, 0.5)
print("Total Loss:", Network.Loss.mse(expects, outputs))
print()

print("模型二：")
net = Network(model_2_json)
print("input = (0.75, 1.25)")
outputs = net.forward((0.75, 1.25))
print("outputs:", outputs)
print("expects = (1)")
expects = 1
print("Total Loss:", Network.Loss.binary_cross_entropy(expects, outputs))
print()
print("input = (-1, 0.5)")
outputs = net.forward((-1, 0.5))
print("outputs:", outputs)
print("expects = (0)")
expects = 0
print("Total Loss:", Network.Loss.binary_cross_entropy(expects, outputs))
print()

print("模型三：")
net = Network(model_3_json)
print("input = (1.5, 0.5)")
outputs = net.forward((1.5, 0.5))
print("outputs:", outputs)
print("expects = (1, 0, 1)")
expects = (1, 0, 1)
print("Total Loss:", Network.Loss.binary_cross_entropy(expects, outputs))
print()
print("input = (0, 1)")
outputs = net.forward((0, 1))
print("outputs:", outputs)
print("expects = (1, 1, 0)")
expects = (1, 1, 0)
print("Total Loss:", Network.Loss.binary_cross_entropy(expects, outputs))
print()

print("模型四：")
net = Network(model_4_json)
print("input = (1.5, 0.5)")
outputs = net.forward((1.5, 0.5))
print("outputs:", outputs)
print("expects = (1, 0, 0)")
expects = (1, 0, 0)
print("Total Loss:", Network.Loss.categorical_cross_entropy(expects, outputs))
print()
print("input = (0, 1)")
outputs = net.forward((0, 1))
print("outputs:", outputs)
print("expects = (0, 0, 1)")
expects = (0, 0, 1)
print("Total Loss:", Network.Loss.categorical_cross_entropy(expects, outputs))
# 互動選擇
while True:
    choice = input("\n請選擇操作 (1: 使用自訂模型, 0: 結束程式): ").strip()
    if choice == "0":
        print("程式結束")
        break
    elif choice == "1":
        # 使用自訂模型
        try:
            network_json = input_network_from_console()
            print("\nGenerated network JSON:\n", network_json)
            # 取得目前日期時間
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 組成檔名
            filename = f"network_JSON_{now_str}.json"

            # 存檔
            with open(filename, "w", encoding="utf-8") as f:
                f.write(network_json)

            print(f"JSON 已經存成 {filename}")

            print("以自訂模型開始運算：")
            net = Network(network_json)
            net.map.show()
            input_values = [
                float(x)
                for x in input("請輸入 input 節點的值，以逗號分隔: ").split(",")
            ]
            outputs = net.forward(input_values)
            print("Forward outputs:", outputs)
        except Exception as e:
            print("發生錯誤:", e)
    else:
        print("輸入錯誤，請輸入 1 或 0")
