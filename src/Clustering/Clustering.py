from PhaseEstimation.vqe import *
from PhaseEstimation import general as qmlgen
import pennylane as qml
import jax
from jax import random
import jax.numpy as jnp
import progressbar
import pickle  # Writing and loading

def get_H_eigvec_operator(mat_H: List[List[int]], N: int, en_lvl: int):
    """
    Function for getting the operator which is transforming |00...0> to the desired eigenvector

    Parameters
    ----------
    qml_H : pennylane.ops.qubit.hamiltonian.Hamiltonian
        Pennylane Hamiltonian of the state
    N : int
        Number of Qubits
    en_lvl : int
        Energy level desired

    Returns
    -------
    np.ndarray
        Operator which transforms |00...0>
    """

    # Compute sorted eigenvalues with jitted function
    eigvals, eigvecs = qmlgen.j_linalgeigh(mat_H)

    psi = eigvecs[:, jnp.argsort(eigvals)[en_lvl]]

    return qml.matrix(qml.QubitStateVector(psi, wires=range(N)))


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


class ClusteringVQE:
    def __init__(self, path_to_file, num_clusters, iterations, show_progress=True):
        self.vqe = load_vqe(path_to_file)
        self.path_to_file = path_to_file
        self.num_clusters = num_clusters
        self.iterations = iterations
        self.clusters = [[] for i in range(num_clusters)]
        self.mean_params = None

        # For Progress
        self.progress = 0
        self.show_progress = show_progress
        self.widgets = [' [', progressbar.Timer(format='elapsed time: %(elapsed)s'), '] ', progressbar.AnimatedMarker(),
                        progressbar.Bar('*'), ' (',
                        progressbar.ETA(), ') ', 'CLusters: ',
                        progressbar.FormatLabel(str([len(cluster) for cluster in self.clusters]))]
        self.bar = progressbar.ProgressBar(maxval=self.iterations * len(self.vqe.vqe_params0),
                                           widgets=self.widgets)

        self.eigenvecs_op = jnp.array(
            [get_H_eigvec_operator(jnp.real(qml.matrix(H)).astype(jnp.float64), self.vqe.Hs.N, 0) for H in
             self.vqe.Hs.qml_Hs])

        @qml.qnode(self.vqe.device, interface="jax")
        def fidelity_vqe(params_phi, params_psi):
            self.vqe.circuit(params_psi)
            qml.adjoint(self.vqe.circuit)(params_phi)
            return qml.probs(wires=range(self.vqe.Hs.N))  # Only interested in |000...0>

        self.v_fidelity_vqe = jax.vmap(
            lambda phi, psi: fidelity_vqe(phi, psi)[0], in_axes=(0, 0)
        )  # vmap of the state circuit
        self.jv_fidelity_vqe = jax.jit(
            self.v_fidelity_vqe
        )

        @qml.qnode(self.vqe.device, interface="jax")
        def fidelity_true_circuit(op_phi, op_psi):
            qml.QubitUnitary(op_phi, wires=range(self.vqe.Hs.N))
            qml.adjoint(qml.QubitUnitary)(op_psi, wires=range(self.vqe.Hs.N))
            return qml.probs(wires=[i for i in range(self.vqe.Hs.N)])  # Only interested in |000...0>

        self.v_fidelity_true = jax.vmap(
            lambda phi, psi: fidelity_true_circuit(phi, psi)[0], in_axes=(0, 0)
        )  # vmap of the state circuit

        self.jv_fidelity_true = jax.jit(
            self.v_fidelity_true
        )

    def compute_clusters(self, centroids: jnp.ndarray, states: jnp.ndarray) -> List[List[int]]:
        clusters = [[] for _ in centroids]

        if len(centroids) > 1:
            for state_index in range(len(states)):
                # index = jnp.argmax(
                #     self.jv_fidelity_vqe(centroids, jnp.array([states[state_index] for _ in range(len(centroids))])))
                index = jnp.argmax(
                    self.jv_fidelity_true(centroids, jnp.array([states[state_index] for _ in range(len(centroids))])))
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

    def cluster(self):
        # Initialize centroids randomly
        # centroids = random.choice(random.PRNGKey(0), self.vqe.vqe_params0, (self.num_clusters,))
        centroids = random.choice(random.PRNGKey(0), self.eigenvecs_op, (self.num_clusters,))
        # indeces = [find_nearest_state(self.vqe.Hs, jnp.array([0, 1.5, -0.5])),
        #            find_nearest_state(self.vqe.Hs, jnp.array([0, 0.2, -0.125])),
        #            find_nearest_state(self.vqe.Hs, jnp.array([0, 0.2, -0.8])), ]
        # centroids = self.vqe.vqe_params0[indeces]
        if self.show_progress:
            self.progress = 0
            self.bar.start()

        for i in range(self.iterations):
            # self.clusters = self.compute_clusters(centroids, self.vqe.vqe_params0)
            self.clusters = self.compute_clusters(centroids, self.eigenvecs_op)
            self.mean_params = compute_mean(self.clusters, self.vqe.Hs)
            # centroids = compute_new_centroids_from_existing_states(self.mean_params, self.vqe.Hs, self.vqe.vqe_params0)
            centroids = compute_new_centroids_from_existing_states(self.mean_params, self.vqe.Hs, self.eigenvecs_op)

        if self.show_progress:
            self.bar.finish()

    def save(self, filename):
        if not isinstance(filename, str):
            raise TypeError("Invalid name for file")

        things_to_save = [
            self.clusters,
            self.mean_params,
            self.path_to_file,
            self.num_clusters,
            self.iterations,
        ]

        with open(filename, "wb") as f:
            pickle.dump(things_to_save, f)


def load(filename):
    if not isinstance(filename, str):
        raise TypeError("Invalid name for file")

    with open(filename, "rb") as f:
        things_to_load = pickle.load(f)

    clusters, mean_params, path_to_file, num_clusters, iterations = things_to_load

    loaded_clustering = ClusteringVQE(path_to_file, num_clusters, iterations)
    loaded_clustering.clusters = clusters
    loaded_clustering.mean_params = mean_params
    return loaded_clustering
