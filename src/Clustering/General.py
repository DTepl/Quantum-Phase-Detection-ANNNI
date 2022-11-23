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

    return psi


def compute_eigenvecs(hamiltonians: qml.ops.qubit.hamiltonian.Hamiltonian, big=False, compute_every=100):
    matr = []

    res = jnp.array([]).reshape(0, 2**hamiltonians.N)

    v_get_op = jax.vmap(
        lambda matrix: get_H_eigvec_operator(matrix, hamiltonians.N, 0), in_axes=(0)
    )
    jv_get_op = jax.jit(v_get_op)

    progress = 0

    widgets = [' [', progressbar.Timer(format='elapsed time: %(elapsed)s'), '] ', progressbar.AnimatedMarker(),
               progressbar.Bar('*'), ' (', progressbar.ETA(), ') ']
    bar = progressbar.ProgressBar(maxval=2 * len(hamiltonians.qml_Hs),
                                  widgets=widgets)

    bar.start()

    for i, qml_H in enumerate(hamiltonians.qml_Hs, 1):
        matr.append(jnp.real(qml.matrix(qml_H).astype(jnp.complex64)).astype(jnp.single))
        progress += 1
        bar.update(progress)

        if big & (i%compute_every == 0):
            res = jnp.concatenate((res, jv_get_op(jnp.array(matr))))
            matr = []
            progress += compute_every
            bar.update(progress)

    if len(matr) != 0:
        res = jnp.concatenate((res, jv_get_op(jnp.array(matr))))
        progress += len(matr)
        bar.update(progress)

    things_to_save = [res, hamiltonians.model_params]

    with open("../../data/clustering/an_eigvecs/N" + str(hamiltonians.N) + "n" + str(
            int(jnp.sqrt(hamiltonians.n_states))), "wb") as f:
        pickle.dump(things_to_save, f)

    bar.finish()
