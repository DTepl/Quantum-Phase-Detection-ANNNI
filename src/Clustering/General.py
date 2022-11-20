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


def compute_eigenvecs(hamiltonians: qml.ops.qubit.hamiltonian.Hamiltonian):
    matr = []
    progress = 0
    widgets = [' [', progressbar.Timer(format='elapsed time: %(elapsed)s'), '] ', progressbar.AnimatedMarker(),
               progressbar.Bar('*'), ' (', progressbar.ETA(), ') ']
    bar = progressbar.ProgressBar(maxval=2 * len(hamiltonians.qml_Hs),
                                  widgets=widgets)

    bar.start()

    for qml_H in hamiltonians.qml_Hs:
        matr.append(jnp.real(qml.matrix(qml_H)))
        progress += 1
        bar.update(progress)

    v_get_op = jax.vmap(
        lambda matr: get_H_eigvec_operator(matr, hamiltonians.N, 0), in_axes=(0)
    )
    jv_get_op = jax.jit(v_get_op)

    res = jv_get_op(jnp.array(matr))
    progress += len(hamiltonians.qml_Hs)
    bar.update(progress)

    things_to_save = [res, hamiltonians.model_params]

    with open("../../data/clustering/an_eigvecs/N" + str(hamiltonians.N) + "n" + str(
            int(jnp.sqrt(hamiltonians.n_states))), "wb") as f:
        pickle.dump(things_to_save, f)

    bar.finish()
