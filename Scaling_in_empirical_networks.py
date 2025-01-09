import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

"""
数据集来源：https://static-content.springer.com/esm/art%3A10.1038%2Fs41567-023-02330-x/MediaObjects/41567_2023_2330_MOESM2_ESM.csv
"""

# 读取CSV文件
data = pd.read_csv('41567_2023_2330_MOESM2_ESM.csv', delimiter=';')

# 提取包含边数和点数信息的网络数据
valid_data = data.dropna(subset=['nodes', 'edges'])

# 计算log(edges)和log(nodes)
valid_data['log_edges'] = np.log10(valid_data['edges'])
valid_data['log_nodes'] = np.log10(valid_data['nodes'])

# 计算γ值
valid_data['gamma'] = valid_data['log_edges'] / valid_data['log_nodes']

# 获取网络类型的种类
network_domains = valid_data['networkDomain'].unique()

# 设置图形风格为无网格白色背景
plt.style.use('classic')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False

# 定义颜色映射
cmap = cm.get_cmap('tab10')

# 绘制散点图，用不同颜色表示不同网络类型，散点半透明且带黑色边框，尺寸变大
for i, domain in enumerate(network_domains):
    domain_data = valid_data[valid_data['networkDomain'] == domain]
    plt.scatter(domain_data['log_nodes'], domain_data['log_edges'], label=domain, alpha=0.5, edgecolors='black', s=50,
                c=[cmap(i)])

# 拟合直线
fit = np.polyfit(valid_data['log_nodes'], valid_data['log_edges'], 1)
fit_fn = np.poly1d(fit)
plt.plot(valid_data['log_nodes'], fit_fn(valid_data['log_nodes']), 'k')

# 设置坐标轴标签
plt.xlabel('log(nodes)')
plt.ylabel('log(edges)')
plt.title('Network Nodes vs Edges by Network Domain')

# 在左上角显示类别颜色标注（背景透明，无边框，字体缩小）
handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0, 1), ncol=2, framealpha=0,
                    fontsize='small')

# 在右上角显示整体γ值
gamma_mean = np.mean(valid_data['gamma'])
gamma_std = np.std(valid_data['gamma'])
plt.text(0.95, 0.95, f'γ={gamma_mean:.2f}±{gamma_std:.2f}', transform=plt.gca().transAxes, ha='right', va='top')

# 创建子图布局，一起显示六个小的散点图
fig, axs = plt.subplots(2, 3, figsize=(10, 8))
axs = axs.flatten()

for i, domain in enumerate(network_domains):
    domain_data = valid_data[valid_data['networkDomain'] == domain]
    axs[i].scatter(domain_data['log_nodes'], domain_data['log_edges'], label=domain, alpha=0.5, edgecolors='black', s=50,
                   c=[cmap(i)])
    fit = np.polyfit(domain_data['log_nodes'], domain_data['log_edges'], 1)
    fit_fn = np.poly1d(fit)
    axs[i].plot(domain_data['log_nodes'], fit_fn(domain_data['log_nodes']), 'k')
    axs[i].set_title(domain)
    axs[i].set_xlabel('log(nodes)')
    axs[i].set_ylabel('log(edges)')

    # 计算每个小散点图对应的γ值
    gamma_mean_domain = np.mean(domain_data['gamma'])
    gamma_std_domain = np.std(domain_data['gamma'])
    axs[i].text(0.95, 0.05, f'γ={gamma_mean_domain:.2f}±{gamma_std_domain:.2f}', transform=axs[i].transAxes, ha='right',
                va='bottom')

    axs[i].legend()

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()