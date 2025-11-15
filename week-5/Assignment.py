import json
import math
from typing import Sequence, List
from datetime import datetime


class Network:

    # ================================================================
    #  一、Activation Functions
    # ================================================================
    class Activation:
        """定義常見激勵函數"""

        def __init__(self):
            pass

        @staticmethod
        def linear(x: Sequence[float]) -> List[float]:
            return x

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
            node_toe: str = "I",
            layer_idx: int = None,
            node_idx: int = None,
        ):
            self.output = 0.0  # activation 後的值
            self.id = self._generate_id(node_toe, layer_idx, node_idx)
            self.pre_layer = pre_layer
            self.curr_layer: "Network._Layer" = None
            self.next_layer: "Network._Layer" = None
            self.weights = list(weights or [])
            self._weight_mapping = {}
            self._d_TL_d_raw = None
            self._gradient_mapping = {}

            # 若有前一層與權重，建立對應表
            if self.pre_layer and self.weights:
                if len(self.pre_layer.nodes) != len(self.weights):
                    raise ValueError(f"{self.id}: 前一層節點數與權重數量不一致")
                self._weight_mapping = dict(zip(self.pre_layer.nodes, self.weights))

        # ------------------------------------------------------------
        @staticmethod
        def _generate_id(node_toe, layer_idx, node_idx):
            """自動生成節點 ID"""
            if node_toe == "I":
                return f"I{node_idx + 1}"
            if node_toe == "H":
                return f"H{(layer_idx or 0) + 1}_{node_idx + 1}"
            if node_toe == "B":
                return f"B{(0 if layer_idx == None else layer_idx + 1)}"
            if node_toe == "O":
                return f"O{node_idx + 1}"
            return f"N{node_idx}"

        # ------------------------------------------------------------
        @property
        def raw_output(self):
            """取得 activation 前的輸出（線性加權結果）"""
            if self.id.startswith("B"):  # Bias node
                return 1.0
            if self.id.startswith("I"):  # Input node
                return self.output
            return sum(
                self._weight_mapping[pre_node] * pre_node_output
                for pre_node, pre_node_output in zip(
                    self.pre_layer.nodes, self.pre_layer.outputs
                )
            )

        @property
        def gradient_mapping(self):
            """取得梯度"""
            return {
                pre_node: self.d_TL_d_raw * pre_node.output
                for pre_node in self.pre_layer
            }

        @gradient_mapping.setter
        def gradient_mapping(self, value):
            """初始化output梯度"""
            self._gradient_mapping = value

        @property
        def d_activation(self):
            """對應激勵函數的導數"""
            act_input = (
                self.output
                if self.curr_layer.activation == "sigmoid"
                else self.raw_output
            )
            func = getattr(
                Network.Activation,
                self.curr_layer.activation + "_derivative",
                Network.Activation.linear_derivative,
            )
            return func(act_input)

        @property
        def d_TL_d_raw(self):
            """Total Loss 對 node raw_output的梯度"""
            if self.id.startswith("O"):
                return self._d_TL_d_raw
            else:
                sum_of_next_node_value = sum(
                    next_node.d_TL_d_raw * next_node._weight_mapping[self]
                    for next_node in self.next_layer
                    if not next_node.id.startswith("B")
                )
                return self.d_activation * sum_of_next_node_value

        @d_TL_d_raw.setter
        def d_TL_d_raw(self, value):
            self._d_TL_d_raw = value

    # ================================================================
    #  四、層（Layer）
    # ================================================================
    class _Layer:
        """神經網路中的一層（input, hidden, output）"""

        def __init__(
            self, nodes: List["Network._Node"], layer_toe: str, activation: str
        ):
            self.nodes = nodes
            self.toe = layer_toe
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
            input_setting, pre_layer=None, layer_toe="I"
        )

        pre_layer = self.map.inputs

        # 建立 hidden layers
        for layer_idx, layer_cfg in enumerate(config.get("layer", [])):
            layer = self._build_layer(
                layer_cfg, pre_layer, layer_toe="H", layer_idx=layer_idx
            )
            self.map.hidden.append(layer)
            pre_layer = layer

        # 建立 output nodes
        output_cfg = config.get("output", {})
        self.map.outputs = self._build_layer(output_cfg, pre_layer, layer_toe="O")

        # 建立 curr_layer, next_layer關係
        for current_layer, next_layer in zip(self.map, self.map[1:] + [None]):
            for node in current_layer:
                node.curr_layer = current_layer
                node.next_layer = next_layer

    # ------------------------------------------------------------
    def _build_layer(
        self,
        cfg: dict,
        pre_layer: "Network._Layer",
        layer_toe: str,
        layer_idx: int = None,
    ) -> "Network._Layer":
        """建立一層（input / hidden / output）"""

        activation = cfg.get("activation", "linear").lower()
        node_count = cfg.get("nodes", 0)

        # input layer
        if layer_toe == "I":
            nodes = [Network._Node(node_toe="I", node_idx=i) for i in range(node_count)]
            nodes.append(Network._Node(node_toe="B"))  # bias node
            return Network._Layer(nodes, layer_toe="I", activation=activation)

        # hidden/output layer
        weights_matrix = cfg.get("weights", [])
        bias_weights = cfg.get("bias_weights", [])
        nodes = []

        for node_idx in range(node_count):
            weights = list(weights_matrix[node_idx]) + [bias_weights[node_idx]]
            node = Network._Node(
                pre_layer=pre_layer,
                weights=weights,
                node_toe=layer_toe,
                layer_idx=layer_idx,
                node_idx=node_idx,
            )
            nodes.append(node)

        # Hidden 層需要 bias node
        if layer_toe != "O":
            nodes.append(
                Network._Node(pre_layer=pre_layer, node_toe="B", layer_idx=layer_idx)
            )
        return Network._Layer(nodes, layer_toe=layer_toe, activation=activation)

    # ------------------------------------------------------------
    def forward(self, input_values: Sequence[float]) -> List[float]:
        """執行前向傳遞（Forward propagation）"""

        input_values = Network._to_list(input_values)
        if len(input_values) != len(self.map.inputs.nodes) - 1:
            raise ValueError("輸入數量與 input node 數量不一致")

        for val, node in zip(input_values, self.map.inputs[:-1]):
            node.output = val

        return self.map.outputs.outputs

    # ------------------------------------------------------------
    def set_output_gradients(
        self,
        output_values: Sequence[float],
        expect_values: Sequence[float],
        loss_func: str,
    ):
        """手動設定outputs d_TL_d_raw 使其自動計算梯度"""
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

            output_node.d_TL_d_raw = (
                d_loss_func(e, o) / (len(output_values) if loss_func == "mse" else 1)
            ) * d_output_layer_activation_func(act_input)

    # ------------------------------------------------------------
    def backward(self) -> str:
        """呼叫 gradient_mapping 使網路其自動計算梯度並回傳梯度字串"""
        parts = ["====NetWork Current Gradient===="]
        for li, layer in enumerate(self.map[1:]):  # skip input layer
            # 判斷層名稱
            if layer.toe == "O":
                layer_name = "Output Layer"
            else:
                layer_name = f"Hidden Layer {li+1}"

            parts.append(f"{layer_name}:")
            for node in layer:
                # 前一層名稱
                pre_layer_name = (
                    "Input Layer"
                    if node.pre_layer.toe == "I"
                    else f"Hidden Layer {self.map.hidden.index(node.pre_layer)+1}"
                )
                grad_dict = {
                    pre_node.id: grad
                    for pre_node, grad in node.gradient_mapping.items()
                }
                parts.append(f"{node.id} <---> {pre_layer_name} {grad_dict}")
        return "\n".join(parts)

    # ------------------------------------------------------------
    def zero_grad(self, learning_rate: float) -> str:
        """更新權重並歸零梯度，回傳更新後的梯度字串"""
        for layer in self.map[1:]:
            for node in layer:
                for pre_node in node._weight_mapping:
                    node._weight_mapping[pre_node] -= (
                        node.gradient_mapping[pre_node] * learning_rate
                    )
                    node.gradient_mapping[pre_node] = 0

        # 建立文字輸出
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
                    pre_node.id: w for pre_node, w in node._weight_mapping.items()
                }
                parts.append(f"  ---> {node.id} {weight_dict}")

        return "\n".join(parts)

    @staticmethod
    def _to_list(x):
        if isinstance(x, (int, float)):
            return [x]
        return list(x)


# ===============================
# 執行
# ===============================
# 第一個模型 JSON 字串
model_1_json = """
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
    },
    {
      "nodes": 1,
      "weights": [
        [
          0.8,
          -0.5
        ]
      ],
      "bias_weights": [
        0.6
      ],
      "activation": "linear"
    }
  ],
  "output": {
    "nodes": 2,
    "weights": [
      [
        0.6
      ],
      [
        -0.3
      ]
    ],
    "bias_weights": [
      0.4,
      0.75
    ],
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

print("程式開始")
print("模型一：")
net = Network(model_1_json)
inputs = (1.5, 0.5)
expects = (0.8, 1)
outputs = net.forward(inputs)
net.set_output_gradients(outputs, expects, "mse")
net.backward()
learning_rate = 0.01

print("=====學習1次=====")
print(net.zero_grad(learning_rate))


print("=====千次學習的Total變化=====")
for i in range(1000):
    outputs = net.forward(inputs)
    net.set_output_gradients(outputs, expects, "mse")
    net.backward()
    net.zero_grad(learning_rate)
    print(f"[{i}]Total Loss:", Network.Loss.mse(expects, outputs))

print()
print("模型二：")
net = Network(model_2_json)
inputs = (0.75, 1.25)
expects = 1
outputs = net.forward(inputs)
net.set_output_gradients(outputs, expects, "binary_cross_entropy")
net.backward()
learning_rate = 0.1

print("=====學習1次=====")
print(net.zero_grad(learning_rate))


print("=====千次學習的Total變化=====")
for i in range(1000):
    outputs = net.forward(inputs)
    net.set_output_gradients(outputs, expects, "binary_cross_entropy")
    net.backward()
    net.zero_grad(learning_rate)
    print(f"[{i}]Total Loss:", Network.Loss.mse(expects, outputs))
