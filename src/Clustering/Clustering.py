from PhaseEstimation.vqe import *
import pennylane as qml
from jax import random
import jax.numpy as jnp
import progressbar
import pickle  # Writing and loading


def compute_mean(clusters: List, hamiltonian: qml.ops.qubit.hamiltonian.Hamiltonian) -> list[jnp.ndarray]:
    cluster_means = []
    for indexes in clusters:
        cluster_means.append(
            jnp.array([0, jnp.mean(hamiltonian.model_params[indexes, 1]),  # 0 as N should be untouched Mean for h
                       jnp.mean(hamiltonian.model_params[indexes, 2])]))  # Mean for K
    return cluster_means


def find_nearest_state(hamiltonian: qml.ops.qubit.hamiltonian.Hamiltonian, mean_arr: jnp.ndarray) -> int:
    diff_array = jnp.abs(hamiltonian.model_params - mean_arr)
    sum_diff_array = jnp.array([row[1] + row[2] for row in diff_array])
    idx = sum_diff_array.argmin()
    return idx


def compute_new_centroids_from_existing_states(mean_params: list[jnp.ndarray],
                                               hamiltonian: qml.ops.qubit.hamiltonian.Hamiltonian, states: jnp.ndarray):
    new_centroids = []
    for cluster_mean in mean_params:
        idx = find_nearest_state(hamiltonian, cluster_mean)
        new_centroids.append(states[idx])
    return new_centroids


class ClusteringVQE:
    def __init__(self, path_to_file, num_clusters, iterations, show_progress=True):
        self.vqe = load_vqe(path_to_file)
        self.num_clusters = num_clusters
        self.iterations = iterations
        self.clusters = [[] for i in range(num_clusters)]

        # For Progress
        self.progress = 0
        self.show_progress = show_progress
        self.widgets = [' [', progressbar.Timer(format='elapsed time: %(elapsed)s'), '] ', progressbar.AnimatedMarker(),
                        progressbar.Bar('*'), ' (',
                        progressbar.ETA(), ') ', 'CLusters: ',
                        progressbar.FormatLabel(str([len(cluster) for cluster in self.clusters]))]
        self.bar = progressbar.ProgressBar(maxval=self.iterations * len(self.vqe.vqe_params0) * self.num_clusters,
                                           widgets=self.widgets)

        @qml.qnode(self.vqe.device, interface="jax")
        def fidelity(params_phi, params_psi):
            qml.adjoint(self.vqe.circuit)(params_phi)
            self.vqe.circuit(params_psi)

            if self.show_progress:
                self.progress += 1
                self.bar.update(self.progress)
            return qml.probs(wires=[i for i in range(self.vqe.Hs.N)])  # Only interested in |000...0>

        self.fidelity = fidelity

    def compute_clusters(self, centroids: jnp.ndarray, states: jnp.ndarray) -> List[List[int]]:
        clusters = [[] for _ in centroids]

        if len(centroids) > 2:
            for state_index in range(len(states)):
                closest, index = self.fidelity(centroids[0], states[state_index])[0], 0
                for i in range(1, len(centroids)):
                    similarity = self.fidelity(centroids[i], states[state_index])[0]
                    if similarity > closest:
                        closest, index = similarity, i
                clusters[index].append(state_index)

                if self.show_progress:
                    self.widgets[len(self.widgets) - 1] = progressbar.FormatLabel(
                        str([len(cluster) for cluster in clusters]))
            return clusters
        else:
            print("There must be at least 2 centroids to cluster!")

    def cluster(self):
        # Initialize centroids randomly
        centroids = random.choice(random.PRNGKey(0), self.vqe.vqe_params0, (self.num_clusters,))
        if self.show_progress:
            self.progress = 0
            self.bar.start()

        for i in range(self.iterations):
            clusters = self.compute_clusters(centroids, self.vqe.vqe_params0)
            mean_params = compute_mean(clusters, self.vqe.Hs)
            centroids = compute_new_centroids_from_existing_states(mean_params, self.vqe.Hs, self.vqe.vqe_params0)
            self.clusters = clusters

        if self.show_progress:
            self.bar.finish()

    def save(self, filename):
        if not isinstance(filename, str):
            raise TypeError("Invalid name for file")

        things_to_save = [
            self.clusters,
            self.num_clusters,
            self.iterations,
        ]

        with open(filename, "wb") as f:
            pickle.dump(things_to_save, f)
