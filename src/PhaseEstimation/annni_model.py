""" This module implements the base function for treating ANNNI model """
import pennylane as qml
from pennylane import numpy as np

##############

def build_H(N, L, K):
    """
    Set up Hamiltonian:
            H = J1* (- Σsigma^i_x*sigma_x^{i+1} - (h/J1) * Σsigma^i_z - (J2/J1) * Σsigma^i_x*sigma_x^{i+2}
        
        [where J1 = 1, (h/J1) = Lambda(/L), (J2/J1) = K]

    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    L : float
        TODO
    K : float
        TODO

    Returns
    -------
    pennylane.ops.qubit.hamiltonian.Hamiltonian
        Hamiltonian Pennylane class for the (Transverse) Ising Chain
    """
    # Interaction of spins with magnetic field
    H = -L * qml.PauliZ(0)
    for i in range(1, N):
        H = H - L * qml.PauliZ(i)

    # Interaction between spins (neighbouring):
    for i in range(0, N - 1):
        H = H + (-1) * (qml.PauliX(i) @ qml.PauliX(i + 1))
        
    # Interaction between spins (next-neighbouring):
    for i in range(0, N - 2):
        print(i, i+2)
        H = H + K * (-1) * (qml.PauliX(i) @ qml.PauliX(i + 2))

    return H

def prepare_Hs(N, n_states):
    """
    Sets up np.ndarray of pennylane Hamiltonians with different parameters
    total_states = n_states * n_states
    Taking n_states values of K from 0 to 1
    Taking n_states values of L from 0 to 2
    
    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    J : float
        Interaction strenght between spins
    n_states : int
        Number of Hamiltonians to generate
    """
    K_states = np.linspace(0, 1, n_states)
    L_states = np.linspace(0, 2, n_states)
    
    Hs = []
    for k in K_states:
        for l in L_states:
            Hs.append(build_H(N, l, k))
        
    return Hs, labels

