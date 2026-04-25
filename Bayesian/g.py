import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

g_true = 9.81
L = 1.0

true_T = 2 * np.pi * np.sqrt(L / g_true)

np.random.seed(42)
T_data = true_T + np.random.normal(0, 0.05, 20)

print("True g:", g_true)
print("True period:", round(true_T, 4))
print("Measured periods:", round(T_data.mean(), 4), "+/-", round(T_data.std(), 4))


g_grid = np.linspace(8, 12, 500)
prior = np.ones_like(g_grid)  # flat prior

def log_likelihood(g):
    T_model = 2 * np.pi * np.sqrt(L / g)
    return np.sum(norm.logpdf(T_data, loc=T_model, scale=0.05))

likelihood = np.exp([log_likelihood(g) for g in g_grid])
posterior = likelihood * prior

posterior /= np.trapz(posterior, g_grid)


plt.plot(g_grid, prior, '--', label='Prior')
plt.plot(g_grid, likelihood, label='Likelihood')
plt.plot(g_grid, posterior, 'k', label='Posterior')
plt.axvline(g_true, color='red', linestyle='--', label='True g')

plt.xlabel('g')
plt.ylabel('Probability')
plt.legend()
plt.show()


g_best = g_grid[np.argmax(posterior)]
print("Best estimate of g:", round(g_best, 3))
