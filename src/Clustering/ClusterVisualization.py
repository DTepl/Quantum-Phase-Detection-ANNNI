from typing import List

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import jax.numpy as jnp
import pennylane as qml

from Clustering import ClusteringVQE, load


def visualize_clusters(clusters: List[List], hamiltonians: qml.ops.qubit.hamiltonian.Hamiltonian):
    plt.figure(figsize=(8, 6), dpi=80)
    phases = mpl.colors.ListedColormap(
        ["lightcoral", "skyblue", "black", "palegreen"]
    )
    norm = mpl.colors.BoundaryNorm(jnp.arange(0, 4), phases.N)
    side = int(jnp.sqrt(hamiltonians.n_states))

    cluster_visual = [i for i in range(hamiltonians.n_states)]

    for i, cluster in enumerate(clusters):
        for index in cluster:
            cluster_visual[index] = i

    sc = plt.imshow(jnp.rot90(jnp.reshape(jnp.array(cluster_visual), (side, side))), cmap=phases, norm=norm)

    plt.ylabel(r"$h$", fontsize=24)
    plt.xlabel(r"$\kappa$", fontsize=24)

    plt.xticks(
        ticks=jnp.linspace(0, side - 1, 5).astype(int),
        labels=[jnp.round(k * 1 / 4, 2) for k in range(0, 5)],
        fontsize=18,
    )
    plt.yticks(
        ticks=jnp.linspace(0, side - 1, 5).astype(int),
        labels=[jnp.round(k * 2 / 4, 2) for k in range(4, -1, -1)],
        fontsize=18,
    )
    plt.colorbar(sc)
    plt.savefig(
        "../../data/clustering/clustering_meaningParams_N" + str(hamiltonians.N) + "n" + str(side) + ".png")
    plt.show()


ClusteringVQEObj = ClusteringVQE("../../data/vqes/ANNNI/N4n100", 3, 5)
ClusteringVQEObj.cluster()
ClusteringVQEObj.save(
   "../../data/clustering/N" + str(ClusteringVQEObj.vqe.Hs.N) + "n" + str(
       int(jnp.sqrt(ClusteringVQEObj.vqe.Hs.n_states))))

# ClusteringVQEObj = load("../../data/clustering/N12n100")
visualize_clusters(ClusteringVQEObj.clusters, ClusteringVQEObj.vqe.Hs)
