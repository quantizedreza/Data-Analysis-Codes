import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

np.random.seed(42)
X = np.sort(np.random.rand(20) * 10).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 20)

X_test = np.linspace(0, 10, 300).reshape(-1, 1)
order = np.random.permutation(len(X))

fig, ax = plt.subplots(figsize=(9, 5))

def update(n):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)
    
    n = max(1, n)
    idx = order[:n]
    gp = GaussianProcessRegressor(RBF(1.0) + WhiteKernel(0.1), optimizer=None)
    gp.fit(X[idx], y[idx])
    
    mu, std = gp.predict(X_test, return_std=True)
    
    ax.plot(X_test, mu, 'r-', lw=2)
    ax.fill_between(X_test.ravel(), mu-1.96*std, mu+1.96*std, color='r', alpha=0.2)
    ax.scatter(X[idx], y[idx], c='k', s=30)
    ax.plot(X_test, np.sin(X_test), 'g--', alpha=0.6)
    ax.set_title(f'Gaussian Process • {n}/{len(X)} points')
    ax.grid(True, alpha=0.3)

ani = FuncAnimation(fig, update, frames=len(X)+1, interval=400, repeat=True)
ani.save('gp_animation.gif', writer='pillow')
plt.show()
