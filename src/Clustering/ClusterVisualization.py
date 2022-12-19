from typing import List, Callable

import matplotlib as mpl
import jax.numpy as jnp
import pennylane as qml

from multiprocessing import Pool
from matplotlib import pyplot as plt
from Clustering import ClusteringVQE, load, Mode
from PhaseEstimation import general as qmlgen


def getlines(
        func: Callable, xrange: List[float], side: int, color: str, res: int = 100
):
    """
    Plot function func from xrange[0] to xrange[1]
    """
    xs = jnp.linspace(xrange[0], xrange[1], res)
    ys = func(xs)
    plt.plot(side * xs - 0.5, side - ys * side / 2 - 0.5, color=color, alpha=0.8)


def visualize_clusters(clusters: List[List], hamiltonians: qml.ops.qubit.hamiltonian.Hamiltonian, mode: Mode,
                       morelines=False):
    plt.figure(figsize=(8, 6), dpi=80)
    phases = mpl.colors.ListedColormap(
        ["lightcoral", "orange", "palegreen", "yellow", "skyblue"]
    )
    norm = mpl.colors.BoundaryNorm(jnp.arange(0, len(clusters) + 1), phases.N)
    side = int(jnp.sqrt(hamiltonians.n_states))

    cluster_visual = [0] * hamiltonians.n_states

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

    getlines(qmlgen.paraanti, [0.5, 1], side, "black", res=100)
    getlines(qmlgen.paraferro, [0, 0.5], side, "black", res=100)
    if morelines:
        getlines(qmlgen.peshel_emery, [0, 0.5], side, "cyan", res=100)
        getlines(qmlgen.b1, [0.5, 1], side, "blue", res=100)

    plt.colorbar(sc)
    plt.savefig(
        "../../data/clustering/figures/clustering_meaningParams_N" + str(
            hamiltonians.N) + "n" + str(side) + "c" + str(len(clusters)) + "m" + str(mode.value) + ".png")
    plt.show()


def run_cluster_procedure(N: int, n: int, max_iterations, mode: Mode, threshold=0.001, visualize=True):
    clustering_vqe_obj = ClusteringVQE("../../data/vqes/ANNNI/N" + str(N) + "n" + str(n), 3, max_iterations,
                                       mode=mode)
    clustering_vqe_obj.cluster(threshold=threshold)
    clustering_vqe_obj.save(
        "../../data/clustering/clusters/N" + str(clustering_vqe_obj.vqe.Hs.N) + "n" + str(
            int(jnp.sqrt(clustering_vqe_obj.vqe.Hs.n_states))) + "c" + str(
            clustering_vqe_obj.num_clusters) + "m" + str(
            mode.value))

    if visualize:
        visualize_clusters(clustering_vqe_obj.clusters, clustering_vqe_obj.vqe.Hs, mode)

    accuracy = clustering_vqe_obj.compute_accuracy()
    print("Accuracy of clustering: " + str(accuracy))
    return accuracy


def load_cluster_visualize(N: int, n: int):
    clustering_vqe_obj = load("../../data/clustering/clusters/N" + str(N) + "n" + str(n) + "c3")
    visualize_clusters(clustering_vqe_obj.clusters, clustering_vqe_obj.vqe.Hs)


def compute_statistics(N: int, n: int, mode: Mode, iterations: int):
    clustering_vqe_obj = ClusteringVQE("../../data/vqes/ANNNI/N" + str(N) + "n" + str(n), 3, 50, mode=mode,
                                       show_progress=False)

    accuracies = []
    for i in range(iterations):
        clustering_vqe_obj.cluster(threshold=0.005)
        accuracies.append(clustering_vqe_obj.compute_accuracy())
    accuracies = jnp.array(accuracies)

    out = "Statistics for N=" + str(N) + " and n=" + str(
        n) + " after " + str(iterations) + " iterations" + ":\nMode - " + mode.name + "\nMean accuracy - " + str(
        jnp.mean(accuracies)) + "\nStandard Deviation - " + str(jnp.std(accuracies))

    with open("../../data/clustering/statistics/N" + str(N) + "n" + str(n) + "_statistics.txt", "w") as file:
        # Writing data to a file
        file.write(out)

    print(out)


def compute_statistics_parallel():
    args = [(4, 100, Mode.analytical, 200), (6, 10, Mode.analytical, 200), (6, 100, Mode.analytical, 200),
            (8, 100, Mode.analytical, 200), (10, 100, Mode.analytical, 200), (12, 100, Mode.analytical, 200)]

    pool = Pool(processes=len(args))
    pool.starmap_async(compute_statistics, args)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # compute_statistics_parallel()
    run_cluster_procedure(10, 100, 50, Mode.vqe, threshold=0.005)
