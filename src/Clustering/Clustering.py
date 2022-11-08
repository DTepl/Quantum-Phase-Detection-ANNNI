from PhaseEstimation.vqe import *
import pennylane as qml
from pennylane import numpy as np
from jax import random
import jax.numpy as jnp

vqe = load_vqe("../../data/vqes/ANNNI/N6n10")

num_centr = 3
centroids = random.choice(random.PRNGKey(0), vqe.vqe_params0, (num_centr,))


@qml.qnode(vqe.device, interface="jax")
def fidelity(params_phi, params_psi):
    qml.adjoint(vqe.circuit)(params_phi)
    vqe.circuit(params_psi)
    return qml.probs(wires=[i for i in range(vqe.Hs.N)])  # Only interested in |000...0>


def compute_clusters(centroids: jnp.ndarray, states):
    clusters = [[] for _ in centroids]

    if (len(centroids) > 2):
        prog = 1
        for state_index in range(len(states)):
            closest, index = fidelity(centroids[0], states[state_index])[0], 0
            for i in range(1, len(centroids)):
                distance = fidelity(centroids[i], states[state_index])[0]
                if distance < closest:
                    closest, index = distance, i
            clusters[index].append(state_index)
            print(prog / len(states))
            prog += 1
        return clusters
    else:
        print("There must be at least 2 centroids to cluster!")


def compute_mean(clusters: List, hamiltonian: qml.ops.qubit.hamiltonian.Hamiltonian):
    cluster_means = []
    for indexes in clusters:
        cluster_means.append(
            [0, jnp.mean(hamiltonian.model_params[indexes, 1]),  # 0 as J should be untouched Mean for h
             jnp.mean(hamiltonian.model_params[indexes, 2])])  # Mena for K
    return cluster_means


def compute_new_centroids(mean_params):
    return


def find_nearest_state(hamiltonian: qml.ops.qubit.hamiltonian.Hamiltonian, mean_arr: jnp.ndarray):
    diff_array = jnp.abs(hamiltonian.model_params - mean_arr)
    sum_diff_array = jnp.array([row[1] + row[2] for row in diff_array])
    idx = sum_diff_array.argmin()
    return idx


clusters = compute_clusters(centroids, vqe.vqe_params0)
mean = compute_mean(clusters, vqe.Hs)
for arr in clusters:
    print(len(arr))
print(mean)
