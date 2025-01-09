from __future__ import division
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

font = {'family': 'arial', 'weight': 'normal'}
plt.rc('font', **font)

SMALL_SIZE = 12
MEDIUM_SIZE = 13
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title


def thermo_spectrum(Ls, beta):
    """
    inputs : the Laplacian spectrum Ls, the propagation scale beta (also indicated as tau)
    outputs : the change in entropy (dS), free energy (dF) and the eta of network formation
    """
    N = len(Ls)
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
    outputs : lists indicating the change in entropy (dS), free energy (dF) and the eta of network formation,
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


# 生成具有社区结构的随机网络（Stochastic Block Model）
def generate_sbm_network(n, sizes, p_in, p_out):
    """
    Generate a Stochastic Block Model network.

    n: number of nodes
    sizes: list of community sizes
    p_in: intra-community connection probability
    p_out: inter-community connection probability
    """
    p_matrix = np.full((len(sizes), len(sizes)), p_out)
    np.fill_diagonal(p_matrix, p_in)
    G = nx.stochastic_block_model(sizes, p_matrix)
    return G


# Parameters for the SBM
N = 1000  # total number of nodes
k = 10  # average degree
num_communities = 10
sizes = [N // num_communities] * num_communities  # evenly sized communities

# Varying mixing parameter mu = k_out / (k_in + k_out)
# For SBM, calculate k_in and k_out based on k and mu
mu_list = np.logspace(-2, 0, 300)  # Varying mixing parameter mu between 0.01 and 1
networks_eta = []
networks_Ls = []

for mu in tqdm(mu_list):
    k_out = k * mu  # External degree (connections between communities)
    k_in = k - k_out  # Internal degree (connections within a community)

    # Generate SBM network
    G = generate_sbm_network(N, sizes, k_in / N, k_out / N)

    # Compute Laplacian spectrum
    networks_Ls.append(np.sort(nx.laplacian_spectrum(G)))

# Propagation scales (beta values)
taus = np.geomspace(10 ** -1, 10 ** 2, 300)

# Calculate eta for each SBM network
for Ls in networks_Ls:
    eta_ = thermo_trajectory(Ls, taus)[-1]
    networks_eta.append(eta_)

networks_eta = np.array(networks_eta).T

# Plot results
plt.figure(figsize=[7, 12], layout='tight')

# Heatmap of eta values across different mixing parameters and propagation scales
x, y = np.meshgrid(np.log(mu_list), np.log(taus))
plt.subplot(2, 1, 1)
plt.title('Order')
plt.pcolormesh(x, y, networks_eta, cmap='viridis')
plt.xticks([])
plt.ylabel(r'$\log{\tau}$')

# Line plot showing the change of eta with different propagation scales (tau)
I = list(np.linspace(70, 280, 7))
I = [int(x) for x in I]
cmap = plt.get_cmap('viridis_r')
slicedCM = cmap(np.linspace(0.1, .9, len(I)))
color_dict = {x: slicedCM[I.index(x)] for x in I}

plt.subplot(2, 1, 2)
for i in I:
    tau = taus[i]
    plt.plot(np.log(mu_list), networks_eta[i, :], label=r'$\tau$=' + str(round(tau, 1)), color=color_dict[i], linewidth=2)

plt.legend(loc='upper left', fancybox=True, framealpha=0.3)
plt.ylim(-0.02, 1)
plt.ylabel(r'$\eta$')
plt.xlabel(r'$\log{\mu}$')
plt.show()
