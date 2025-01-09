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

def scale_free(n, m, alpha):
    """
    inputs: number of nodes n, barabasi-albert parameter m, the probability of reversing the "rich get richer" law
    output: NetworkX graph

    """
    if m < 1 or m >= n:
        raise nx.NetworkXError("Preferential attactment algorithm must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))
        # Add m initial nodes (m0 in barabasi-speak)
    G = nx.empty_graph(m)

    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)

        P = []
        for i in range(max(repeated_nodes)):
            P.append(repeated_nodes.count(i))
        P = np.array(P) / sum(P)

        Q = P
        Q[np.where(Q == 0)] = 1
        Q = 1 / (Q ** 10)
        Q = Q / sum(Q)

        P = (1 - alpha) * P + alpha * Q

        targets = []
        rep = []
        for i in range(m):
            tar = np.random.choice(np.arange(0, len(P)), p=P)
            while tar in rep:
                tar = np.random.choice(np.arange(0, len(P)), p=np.array(P) / sum(P))
            rep.append(tar)
            targets.append(tar)
        source += 1
    return G

N = 500 # number of nodes
m = 3
alpha_list = np.geomspace(0.1/N, 0.8, 200) #list of connectivity probabilities
networks_Ls = []  #list of ER network spectrums with different connectivity probabilities
for alpha in tqdm(alpha_list):
    G = scale_free(N,m,alpha)
    networks_Ls.append(np.sort(nx.laplacian_spectrum(G)))

networks_eta = []
taus = np.geomspace(10 ** -1, 10 ** 2, 300)

for Ls in networks_Ls:
    eta_ = thermo_trajectory(Ls, taus)[-1]
    networks_eta.append(eta_)

networks_eta = np.array(networks_eta).T

plt.figure(figsize=[7, 12], layout='tight')

x, y = np.meshgrid(np.log(alpha_list), np.log(taus))
plt.subplot(2, 1, 1)
plt.title('Order')
plt.pcolormesh(x, y, networks_eta, cmap='viridis')
plt.xticks([])
plt.ylabel(r'$\log{\tau}$')

I = list(np.linspace(50, 250, 7))
I = [int(x) for x in I]
cmap = plt.get_cmap('viridis_r')
slicedCM = cmap(np.linspace(0.1, .9, len(I)))
color_dict = {x: slicedCM[I.index(x)] for x in I}

plt.subplot(2, 1, 2)
for i in I:
    tau = taus[i]
    plt.plot(np.log(alpha_list), networks_eta[i, :], label=r'$\tau$=' + str(round(tau, 1)), color=color_dict[i], linewidth=2)

plt.legend(loc='upper left', fancybox=True, framealpha=0.3)
# plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center')
plt.ylim(-0.02, 1)
plt.ylabel(r'$\eta$')
plt.xlabel(r'$\log{p_{rev}}$')
plt.show()