## 一、代码依赖库

- **`numpy`**
- **`networkx`**
- **`matplotlib.pyplot`**
- **`tqdm`**
- **`random`**

## 二、代码文件说明

1. **Scaling_in_empirical_networks.py :** 实现了对特定网络数据的读取、处理以及可视化展示操作。通过一系列的数据处理和绘图操作，呈现网络中节点数与边数之间的关系，并按照网络类型进行区分展示相关的拟合直线以及计算出的 `γ` 值等关键信息。
2. **Disconnected_to_connected.py : **生成不同连接概率的 ER 图，分析其在多个传播尺度下的热力学特性。
3. **Random_block_model.py : **生成具有不同混合参数的 SBM 网络，计算其拉普拉斯谱，分析网络在多个传播尺度下的热力学特性。
4. **Disordered_to_ordered.py : **生成不同重连概率下的小世界网络，计算其拉普拉斯谱，进而分析网络在多个传播尺度下的热力学特性。
5. **Modified_BA_model.py : **生成不同反转概率下的无标度网络，计算其拉普拉斯谱，分析网络在多个传播尺度下的热力学特性。