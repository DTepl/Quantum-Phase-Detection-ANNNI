from typing import List, Callable

import matplotlib as mpl
import jax.numpy as jnp
from pennylane import numpy as np
import pennylane as qml

from matplotlib import pyplot as plt
from Clustering import ClusteringVQE, load, Mode
from PhaseEstimation import general as qmlgen

from PhaseEstimation.visualization import getlines
from matplotlib import rc

rc("text", usetex=False)


def visualize_clusters(clusters: List[List], hamiltonians: qml.ops.qubit.hamiltonian.Hamiltonian, morelines=False):
    plt.figure(figsize=(8, 6), dpi=80)
    phases = mpl.colors.ListedColormap(
        ["lightcoral", "skyblue", "black", "palegreen", "yellow"]
    )
    norm = mpl.colors.BoundaryNorm(jnp.arange(0, len(clusters) + 1), phases.N)
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

    getlines(qmlgen.paraanti, [0.5, 1], side, "white", res=100)
    getlines(qmlgen.paraferro, [0, 0.5], side, "white", res=100)
    if morelines:
        getlines(qmlgen.peshel_emery, [0, 0.5], side, "cyan", res=100)
        getlines(qmlgen.b1, [0.5, 1], side, "blue", res=100)

    plt.colorbar(sc)
    plt.savefig(
        "../../data/clustering/figures/clustering_meaningParams_N" + str(
            hamiltonians.N) + "n" + str(side) + "c" + str(len(clusters)) + ".png")
    plt.show()


mode = Mode.quantum_analytical
ClusteringVQEObj = ClusteringVQE("../../data/vqes/ANNNI/N8n100", 3, 5, mode=mode)
ClusteringVQEObj.cluster()
ClusteringVQEObj.save(
    "../../data/clustering/clusters/N" + str(ClusteringVQEObj.vqe.Hs.N) + "n" + str(
        int(jnp.sqrt(ClusteringVQEObj.vqe.Hs.n_states))) + "c" + str(ClusteringVQEObj.num_clusters) + "m" + str(mode.value))

# ClusteringVQEObj = load("../../data/clustering/clusters/N6n100c3")
visualize_clusters(ClusteringVQEObj.clusters, ClusteringVQEObj.vqe.Hs)
