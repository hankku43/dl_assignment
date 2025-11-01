import json
from typing import Iterable, Sequence, List
from datetime import datetime


class Network:
    """簡易forward神經網路"""

    class _node:
        def __init__(
            self,
            pre_nodes: List["Network._Node"] = None,
            weights: Sequence[float] = None,
            node_type: str = "I",
            layer_idx: int = None,
            node_idx: int = None,
        ):
            self._output = 0.0

            # 自動生成節點 ID
            self.id = self._generate_id(node_type, layer_idx, node_idx)

            # 建立前一層節點與權重的對應
            self.pre_nodes = pre_nodes or []
            self.weights = list(weights or [])
            if self.pre_nodes and self.weights:
                if len(self.pre_nodes) != len(self.weights):
                    raise ValueError(f"{self.id}: pre_nodes 與 weights 長度不一致")
                self._weight_mapping = dict(zip(self.pre_nodes, self.weights))
            else:
                self._weight_mapping = {}

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

        @property
        def output(self):
            """取得節點輸出，如果有前一節點(非input)就計算線性組合，如果是bias就直接輸出1"""
            if self.id.startswith("B"):
                return 1.0
            if not self._weight_mapping:
                return self._output
            return sum(self._weight_mapping[p] * p.output for p in self.pre_nodes)

        @output.setter
        def output(self, value):
            self._output = value

    class NetworkMap:
        """用於儲存整體網路結構"""

        def __init__(self):
            self.inputs: List[Network._node] = []
            self.hidden: List[List[Network._node]] = []
            self.outputs: List[Network._node] = []

        def show(self):
            """以文字方式顯示網路結構與連線"""
            print("\n=== Network Structure ===")

            # 顯示輸入層節點
            print("\nInput Layer:")
            print("  " + ", ".join(n.id for n in self.inputs))

            # 顯示隱藏層
            for idx, layer in enumerate(self.hidden, 1):
                print(f"\nHidden Layer {idx}:")
                for node in layer:
                    for pre_node, w in node._weight_mapping.items():
                        print(f"  {pre_node.id} -{w:.2f}-> {node.id}")

            # 顯示輸出層
            print("\nOutput Layer:")
            for node in self.outputs:
                for pre_node, w in node._weight_mapping.items():
                    print(f"  {pre_node.id} -{w:.2f}-> {node.id}")

            print()

    def __init__(self, network_setting_json: str):
        setting = json.loads(network_setting_json)
        self.map = Network.NetworkMap()

        # 建立 input nodes
        input_count = setting.get("input", 0)
        self.map.inputs = [
            Network._node(node_type="I", node_idx=i) for i in range(input_count)
        ] + [Network._node(node_type="B")]

        prev_layer = self.map.inputs

        # 建立 hidden layers
        for layer_idx, layer_setting in enumerate(setting.get("layer", [])):
            nodes = self._build_layer(layer_setting, prev_layer, "H", layer_idx)
            self.map.hidden.append(nodes)
            prev_layer = nodes

        # 建立 output nodes
        output_setting = setting.get("output", {})
        self.map.outputs = self._build_layer(output_setting, prev_layer, "O")

    def _build_layer(
        self,
        setting: dict,
        prev_layer: List["Network._node"],
        node_type: str,
        layer_idx: int = None,
    ) -> List["Network._node"]:
        """共用層生成邏輯（支援 hidden/output）"""
        node_count = setting.get("nodes", 0)
        weights_matrix = setting.get("weights", [])
        bias_weights = setting.get("bias_weights", [])
        nodes = []

        for node_idx in range(node_count):
            weights = list(weights_matrix[node_idx]) + [bias_weights[node_idx]]
            node = Network._node(
                pre_nodes=prev_layer,
                weights=weights,
                node_type=node_type,
                layer_idx=layer_idx,
                node_idx=node_idx,
            )
            nodes.append(node)

        # 非輸出層需要 bias node
        if node_type != "O":
            nodes.append(
                Network._node(pre_nodes=prev_layer, node_type="B", layer_idx=layer_idx)
            )
        return nodes

    def forward(self, input_values: Sequence[float]) -> List[float]:
        if len(input_values) != len(self.map.inputs) - 1:
            raise ValueError("輸入數量與 input node 數量不一致")
        for val, node in zip(input_values, self.map.inputs[:-1]):
            node.output = val
        return [node.output for node in self.map.outputs]


# ===============================
# Console 互動輸入模型設定並生成 JSON
# ===============================
def input_network_from_console():
    input_count = int(input("Input node 數量: "))
    hidden_str = input("Hidden layers 節點數量 (逗號分隔，例如 3,2): ")
    hidden_layers = [int(x) for x in hidden_str.split(",") if x.strip()]
    output_count = int(input("Output node 數量: "))

    network_dict = {
        "input": input_count,
        "layer": [],
        "output": {"nodes": output_count, "weights": [], "bias_weights": []},
    }

    prev_nodes = input_count

    # Hidden layers
    for layer_idx, nodes_count in enumerate(hidden_layers):
        print(f"\n=== 輸入 Hidden Layer {layer_idx+1} 權重 ===")
        weights = []
        for node_idx in range(nodes_count):
            val = input(
                f"請輸入第 {node_idx+1} 個節點權重，用逗號分隔 (對應前一層 {prev_nodes} 個節點): "
            )
            w_list = [float(x) for x in val.split(",")]
            if len(w_list) != prev_nodes:
                raise ValueError(f"權重數量必須是 {prev_nodes}")
            weights.append(w_list)

        # bias weights
        bias_input = input(
            f"請輸入 Hidden Layer {layer_idx+1} 每個節點 bias 權重，用逗號分隔: "
        )
        bias_weights = [float(x) for x in bias_input.split(",")]
        if len(bias_weights) != nodes_count:
            raise ValueError(f"bias 權重數量必須是 {nodes_count}")

        network_dict["layer"].append(
            {"nodes": nodes_count, "weights": weights, "bias_weights": bias_weights}
        )
        prev_nodes = nodes_count

    # Output layer
    print("\n=== 輸入 Output Layer 權重 ===")
    weights = []
    for node_idx in range(output_count):
        val = input(
            f"請輸入第 {node_idx+1} 個輸出節點權重，用逗號分隔 (對應前一層 {prev_nodes} 個節點): "
        )
        w_list = [float(x) for x in val.split(",")]
        if len(w_list) != prev_nodes:
            raise ValueError(f"權重數量必須是 {prev_nodes}")
        weights.append(w_list)

    bias_input = input("請輸入 Output Layer 每個節點 bias 權重，用逗號分隔: ")
    bias_weights = [float(x) for x in bias_input.split(",")]
    if len(bias_weights) != output_count:
        raise ValueError(f"bias 權重數量必須是 {output_count}")

    network_dict["output"]["weights"] = weights
    network_dict["output"]["bias_weights"] = bias_weights

    return json.dumps(network_dict, indent=2)


# ===============================
# 執行
# ===============================
# 第一個模型 JSON 字串
model_1_json = """
{
  "input": 2,
  "layer": [
    {
      "nodes": 2,
      "weights": [
        [0.5, 0.2],
        [0.6, -0.6]
      ],
      "bias_weights": [0.3, 0.25]
    }
  ],
  "output": {
    "nodes": 1,
    "weights": [
      [0.8, 0.4]
    ],
    "bias_weights": [-0.5]
  }
}
"""

# 第二個模型 JSON 字串
model_2_json = """
{
  "input": 2,
  "layer": [
    {
      "nodes": 2,
      "weights": [
        [0.5, 1.5],
        [0.6, -0.8]
      ],
      "bias_weights": [0.3, 1.25]
    },
    {
      "nodes": 1,
      "weights": [
        [0.6, -0.8]
      ],
      "bias_weights": [0.3]
    }
  ],
  "output": {
    "nodes": 2,
    "weights": [
      [0.5],
      [-0.4]
    ],
    "bias_weights": [0.2, 0.5]
  }
}
"""
print("程式開始")
print("模型一：")
net = Network(model_1_json)
print("input = (1.5, 0.5)")
result = net.forward((1.5, 0.5))
print("result:", result)
print()
print("input = (0, 1)")
result = net.forward((0, 1))
print("result:", result)
print()

print("模型二：")
net = Network(model_2_json)
print("input = (0.75, 1.25)")
result = net.forward((0.75, 1.25))
print("result:", result)
print()
print("input = (-1, 0.5)")
result = net.forward((-1, 0.5))
print("result:", result)

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
            net.map.show()

            print("以自訂模型開始運算：")
            net = Network(network_json)
            input_values = [
                float(x)
                for x in input("請輸入 input 節點的值，以逗號分隔: ").split(",")
            ]
            result = net.forward(input_values)
            print("Forward result:", result)
        except Exception as e:
            print("發生錯誤:", e)
    else:
        print("輸入錯誤，請輸入 1 或 0")
