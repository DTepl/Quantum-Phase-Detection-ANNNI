from typing import List

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import jax.numpy as jnp
import pennylane as qml

from src.Clustering.Clustering import ClusteringVQE


def visualize_clusters(clusters: List[List], hamiltonians: qml.ops.qubit.hamiltonian.Hamiltonian):
    plt.figure(figsize=(8, 6), dpi=80)
    phases = mpl.colors.ListedColormap(
        ["lightcoral", "skyblue", "black", "palegreen"]
    )
    norm = mpl.colors.BoundaryNorm(jnp.arange(0, 4), phases.N)

    sc = []

    for i, cluster in enumerate(clusters):
        params = hamiltonians.model_params[cluster]
        h = params[:, 1]
        k = params[:, 2]
        c = [i for _ in range(len(h))]
        sc = plt.scatter(k, h, c=c, cmap=phases, norm=norm)

    plt.ylabel(r"$h$", fontsize=24)
    plt.xlabel(r"$\kappa$", fontsize=24)
    plt.colorbar(sc)
    plt.show()


ClusteringVQEObj = ClusteringVQE("../../data/vqes/ANNNI/N6n10", 3, 5)
ClusteringVQEObj.cluster()
visualize_clusters(ClusteringVQEObj.clusters, ClusteringVQEObj.vqe.Hs)
