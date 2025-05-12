import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import eigsh
from itertools import product

# Parameters
L = 5           # number of sites
N = 4           # total number of bosons
t = 1.0        # hopping amplitude
U = 5.0         # on-site interaction strength

# Generate basis: all integer vectors (n1, n2, ..., nL) with sum = N
def generate_basis(L, N):
    basis = []
    for config in product(range(N+1), repeat=L):
        if sum(config) == N:
            basis.append(config)
    return basis

basis = generate_basis(L, N)
dim = len(basis)
state_index = {state: idx for idx, state in enumerate(basis)}

# Create Hamiltonian (sparse matrix)
H = dok_matrix((dim, dim), dtype=np.float64)

# Fill diagonal (interaction terms)
for idx, state in enumerate(basis):
    interaction_energy = sum(0.5 * U * n * (n - 1) for n in state)
    H[idx, idx] = interaction_energy

# Fill off-diagonal (hopping terms)
for idx, state in enumerate(basis):
    for i in range(L):  # Open boundary; for periodic, add (L-1,0) as well
        j = (i + 1) % L
        n_i, n_j = state[i], state[j]
        if n_i > 0:
            new_state = list(state)
            new_state[i] -= 1
            new_state[j] += 1
            new_state = tuple(new_state)
            if new_state in state_index:
                jdx = state_index[new_state]
                amp = -t * np.sqrt(n_i * (n_j + 1))
                H[idx, jdx] += amp
                H[jdx, idx] += amp  # Hermitian

        if n_j > 0:
            new_state = list(state)
            new_state[j] -= 1
            new_state[i] += 1
            new_state = tuple(new_state)
            if new_state in state_index:
                jdx = state_index[new_state]
                amp = -t * np.sqrt(n_j * (n_i + 1))
                H[idx, jdx] += amp
                H[jdx, idx] += amp  # Hermitian

# Diagonalize
vals, vecs = eigsh(H.tocsr(), k=1, which='SA')  # smallest eigenvalue

# Output ground state
ground_state = vecs[:, 0]
print(f"Ground state energy: {vals[0]:.4f}\n")
print("Wavefunction coefficients:")
for idx, coeff in enumerate(ground_state):
    print(f"{basis[idx]}: {coeff:.4f}")
    
import matplotlib.pyplot as plt

for idx, coeff in enumerate(ground_state):
    plt.bar(idx, abs(coeff)**2, label=str(basis[idx]))
plt.xlabel('Basis State')
plt.ylabel('Probability Amplitude')
plt.title('Ground State Wavefunction Coefficients')
basis_labels = [str(b) for b in basis]
plt.xticks(range(len(basis)) , basis_labels, rotation=90)
plt.tight_layout()
plt.show()
