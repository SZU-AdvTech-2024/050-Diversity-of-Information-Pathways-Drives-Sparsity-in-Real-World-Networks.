from __future__ import division
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from progressbar import ProgressBar
import scipy as sci
import random

from tqdm import tqdm

font = {'family' : 'arial', 'weight' :'normal'}

plt.rc('font', **font)

SMALL_SIZE = 12
MEDIUM_SIZE = 13
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

def thermo_spectrum(Ls, beta):
    """
    inputs : the Laplacian spectrum Ls, the propagation scale beta (also indicated as tau)
    outputs : the change in entropy (dS), free enrgy (dF) and the eta of network formation
    """

    N = len(Ls)
    # Nc = len(np.where(Ls<10**-10)[0])
    p = np.exp(-beta * Ls)
    Ls = np.delete(Ls, np.where(p < 10 ** -12))
    p = np.delete(p, np.where(p < 10 ** -12))
    Z = np.sum(p)
    p = p / Z
    dF = (np.log(N) - np.log(Z))
    dS = np.sum(-p * np.log(p)) - np.log(N)
    eta = (dF + dS) / dF
    return dS, dF, eta


def thermo_trajectory(Ls, beta_list):
    """
    inputs : the Laplacian spectrum Ls, a list of propagation scales beta (also indicated as tau)
    outputs : lists indicating the change in entropy (dS), free enrgy (dF) and the eta of network formation,
    at each propagation scale
    """
    n = len(beta_list)
    dS_ = np.zeros(n)
    dF_ = np.zeros(n)
    eta_ = np.zeros(n)
    for i in range(n):
        beta = beta_list[i]
        dS_[i], dF_[i], eta_[i] = thermo_spectrum(Ls, beta)
    return dS_, dF_, eta_

#  生成小世界网络
def generate_small_world_network(n, k, p_rew):
    return nx.watts_strogatz_graph(n, k, p_rew)

N = 1000  # 假设节点数量为1000，可根据需要调整
k = 8     # 平均度
p_rew_list = np.linspace(0.01, 1.01, 500)  # 重连概率列表，这里取0到1之间的10个值，可根据需要调整
networks_eta = []
networks_Ls = []
for p_rew in p_rew_list:
    G = generate_small_world_network(N, k, p_rew)
    networks_Ls.append(np.sort(nx.laplacian_spectrum(G)))

networks_eta = []
taus = np.geomspace(10 ** -1, 10 ** 2, 300)

for Ls in networks_Ls:
    eta_ = thermo_trajectory(Ls, taus)[-1]
    networks_eta.append(eta_)

networks_eta = np.array(networks_eta).T

plt.figure(figsize=[7, 12], layout='tight')

x, y = np.meshgrid(np.log(p_rew_list), np.log(taus))
plt.subplot(2, 1, 1)
plt.title('Order')
plt.pcolormesh(x, y, networks_eta, cmap='viridis')
plt.xticks([])
plt.ylabel(r'$\log{\tau}$')

I = list(np.linspace(80, 230, 6))
I = [int(x) for x in I]
cmap = plt.get_cmap('viridis_r')
slicedCM = cmap(np.linspace(0.1, .9, len(I)))
color_dict = {x: slicedCM[I.index(x)] for x in I}

plt.subplot(2, 1, 2)
for i in I:
    tau = taus[i]
    plt.plot(np.log(p_rew_list), networks_eta[i, :], label=r'$\tau$=' + str(round(tau, 1)), color=color_dict[i], linewidth=2)

plt.legend(loc='upper left', fancybox=True, framealpha=0.3)
# plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center')
plt.ylim(-0.02, 1)
plt.ylabel(r'$\eta$')
plt.xlabel(r'$\log{p_{rew}}$')
plt.show()