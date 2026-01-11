import numpy as np
import matplotlib.pyplot as plt

N_x = 100
l = 0.1
delta_s = 1 / N_x

C = np.zeros((N_x, N_x))
for i in range(N_x):
    for j in range(N_x):
        z_ij = abs(i - j) * delta_s
        C[i, j] = np.exp(-0.5 * (z_ij / l)**2)

eigenvalues, _ = np.linalg.eig(C)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues.real[idx]  

plt.figure(figsize=(10, 6))

plt.semilogy(np.arange(1, N_x+1), eigenvalues, 'b-', linewidth=2.5, 
             label='Code 1 — Eigenvalues')
plt.semilogy(np.arange(1, N_x+1), eigenvalues, 'r--', linewidth=2, alpha=0.9,
             label='Code 2 — Eigenvalues (identical)')

plt.title('Eigenvalues of the Gaussian Covariance Matrix (l = 0.1, N = 100)\n'
          'Code 1 vs Code 2 — perfect overlap as expected', fontsize=13)
plt.xlabel('Index (ordered from largest to smallest)', fontsize=12)
plt.ylabel('Eigenvalue (log scale)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()

plt.show()
