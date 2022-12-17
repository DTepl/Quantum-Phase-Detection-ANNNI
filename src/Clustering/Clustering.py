from PhaseEstimation.vqe import *
import random
from enum import Enum
import pennylane as qml
from sympy.utilities.iterables import multiset_permutations
import jax
import jax.numpy as jnp
import progressbar
import pickle  # Writing and loading


def compute_mean(clusters: List, hamiltonian: qml.ops.qubit.hamiltonian.Hamiltonian) -> list[jnp.ndarray]:
    cluster_means = []
    for indexes in clusters:
        cluster_means.append(
            jnp.array(
                [0, jax.jit(jnp.mean)(hamiltonian.model_params[indexes, 1]),  # 0 as N should be untouched Mean for h
                 jax.jit(jnp.mean)(hamiltonian.model_params[indexes, 2])]))  # Mean for K
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
    return jnp.array(new_centroids)


class Mode(Enum):
    vqe = 1
    quantum_analytical = 2
    analytical = 3


class ClusteringVQE:

    def __init__(self, path_to_vqe_file, num_clusters, iterations, show_progress=True, mode=Mode.vqe):
        self.vqe = load_vqe(path_to_vqe_file)
        self.path_to_vqe_file = path_to_vqe_file
        self.num_clusters = num_clusters
        self.iterations = iterations
        self.clusters = [[] for _ in range(num_clusters)]
        self.mean_params = None

        if mode == Mode.vqe:
            self.data_points = self.vqe.vqe_params0

            @qml.qnode(self.vqe.device, interface="jax")
            def fidelity(params_phi, params_psi):
                self.vqe.circuit(params_psi)
                qml.adjoint(self.vqe.circuit)(params_phi)
                return qml.probs(wires=range(self.vqe.Hs.N))
        elif mode == Mode.quantum_analytical:
            with open("../../data/clustering/an_eigvecs/N" + str(self.vqe.Hs.N) + "n" + str(
                    int(jnp.sqrt(self.vqe.Hs.n_states))), "rb") as f:
                self.data_points = pickle.load(f)[0]

            @qml.qnode(self.vqe.device, interface="jax")
            def fidelity(state_phi, state_psi):
                qml.MottonenStatePreparation(state_phi, wires=range(self.vqe.Hs.N))
                qml.adjoint(qml.MottonenStatePreparation)(state_psi, wires=range(self.vqe.Hs.N))
                return qml.probs(wires=[i for i in range(self.vqe.Hs.N)])
        elif mode == Mode.analytical:
            with open("../../data/clustering/an_eigvecs/N" + str(self.vqe.Hs.N) + "n" + str(
                    int(jnp.sqrt(self.vqe.Hs.n_states))), "rb") as f:
                self.data_points = pickle.load(f)[0]

            def fidelity(state_phi, state_psi):
                return [jnp.abs(jnp.vdot(state_phi, state_psi))]
        else:
            raise LookupError("You need to specify in which mode to run!")

        self.v_fidelity = jax.vmap(
            lambda phi, psi: fidelity(phi, psi)[0], in_axes=(0, 0)  # Only interested in |000...0>
        )  # vmap of the state circuit
        self.jv_fidelity = jax.jit(
            self.v_fidelity
        )

        # For Progress
        self.progress = 0
        self.show_progress = show_progress
        self.widgets = [' [', progressbar.Timer(format='elapsed time: %(elapsed)s'), '] ', progressbar.AnimatedMarker(),
                        progressbar.Bar('*'), ' (',
                        progressbar.ETA(), ') ', 'CLusters: ',
                        progressbar.FormatLabel(str([len(cluster) for cluster in self.clusters]))]
        self.bar = progressbar.ProgressBar(maxval=self.iterations * len(self.data_points), widgets=self.widgets)

    def compute_clusters(self, centroids: jnp.ndarray, states: jnp.ndarray) -> List[List[int]]:
        clusters = [[] for _ in centroids]

        if len(centroids) > 1:
            for state_index in range(len(states)):
                index = jnp.argmax(
                    self.jv_fidelity(centroids, jnp.array([states[state_index] for _ in range(len(centroids))])))
                clusters[index].append(state_index)

                if self.show_progress:
                    self.progress += 1
                    self.bar.update(self.progress)
                    self.widgets[len(self.widgets) - 1] = progressbar.FormatLabel(
                        str([len(cluster) for cluster in clusters]))
            return clusters
        else:
            print("There must be at least 2 centroids to cluster!")
            exit(1)

    def cluster(self, threshold=0):
        # Initialize centroids randomly
        centroid_indices = random.choices(range(len(self.data_points)), k=self.num_clusters)
        centroids = self.data_points[centroid_indices]
        old_mean_params = self.vqe.Hs.model_params[centroid_indices]

        # indeces = jnp.array([find_nearest_state(self.vqe.Hs, jnp.array([0, 1.5, -0.5])),
        #            find_nearest_state(self.vqe.Hs, jnp.array([0, 0.2, -0.125])),
        #            find_nearest_state(self.vqe.Hs, jnp.array([0, 0.2, -0.8]))])
        # centroids = self.data_points[indeces]

        if self.show_progress:
            self.progress = 0
            self.bar.start()

        for i in range(self.iterations):
            self.clusters = self.compute_clusters(centroids, self.data_points)
            self.mean_params = compute_mean(self.clusters, self.vqe.Hs)
            centroids = compute_new_centroids_from_existing_states(self.mean_params, self.vqe.Hs, self.data_points)

            diff = (old_mean_params - jnp.array(self.mean_params))[:, [1, 2]]
            old_mean_params = jnp.array(self.mean_params.copy())

            if (diff <= threshold).all():
                break

        if self.show_progress:
            self.bar.finish()

    def compute_accuracy(self):
        side = jnp.sqrt(self.vqe.Hs.n_states)
        permutations = list(multiset_permutations([[0, 1], [1, 0], [1, 1]]))
        predictions = [[0] * self.vqe.Hs.n_states for _ in permutations]

        for i, cluster in enumerate(self.clusters):
            for index in cluster:
                for x, permutation in enumerate(permutations):
                    predictions[x][index] = permutation[i]

        # Compare predictions to actual states
        # applying inequalities to theoretical curves
        labels = []
        for idx in range(self.vqe.Hs.n_states):
            # compute coordinates and normalize for x in [0,1]
            # and y in [0,2]
            x = (idx // side) / side
            y = 2 * (idx % side) / side

            # If x==0 we get into 0/0 on the theoretical curve
            if x == 0:
                if 1 <= y:
                    labels.append([1, 1])
                else:
                    labels.append([0, 1])
            elif x <= 0.5:
                if qmlgen.paraferro(x) <= y:
                    labels.append([1, 1])
                else:
                    labels.append([0, 1])
            else:
                if (qmlgen.paraanti(x)) <= y:
                    labels.append([1, 1])
                else:
                    labels.append([1, 0])

        correct = jnp.sum(jnp.array(labels) == jnp.array(predictions), axis=2).astype(int) == 2
        accuracy = (jnp.sum(correct, axis=1) / (side * side)).max()
        return accuracy

    def save(self, filename):
        if not isinstance(filename, str):
            raise TypeError("Invalid name for file")

        things_to_save = [
            self.clusters,
            self.mean_params,
            self.path_to_vqe_file,
            self.num_clusters,
            self.iterations,
        ]

        with open(filename, "wb") as f:
            pickle.dump(things_to_save, f)


def load(filename, mode=Mode.vqe):
    if not isinstance(filename, str):
        raise TypeError("Invalid name for file")

    with open(filename, "rb") as f:
        things_to_load = pickle.load(f)

    clusters, mean_params, path_to_vqe_file, num_clusters, iterations = things_to_load

    loaded_clustering = ClusteringVQE(path_to_vqe_file, num_clusters, iterations, mode=mode)
    loaded_clustering.clusters = clusters
    loaded_clustering.mean_params = mean_params
    return loaded_clustering
