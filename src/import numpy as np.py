import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Candidate parameter grid
mus = np.arange(3.8, 5.0, 0.2)   # [3.8, 4.0, 4.2, 4.4, 4.6, 4.8]
sigmas = np.arange(0.3, 0.8, 0.1)  # [0.3, 0.4, 0.5, 0.6, 0.7]

x = np.linspace(0, 600, 1000)  # service time axis in minutes (adjust upper limit as needed)

# Plot grid of PDFs
fig, axes = plt.subplots(len(sigmas), len(mus), figsize=(15, 10), sharex=True, sharey=True)

for i, sigma in enumerate(sigmas):
    for j, mu in enumerate(mus):
        ax = axes[i, j]
        
        # scipy lognorm parameterisation: shape = sigma, scale = exp(mu)
        pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
        ax.plot(x, pdf, color="blue")
        
        ax.set_title(f"μ={mu:.1f}, σ={sigma:.1f}", fontsize=8)
        ax.set_xlim(0, 600)  # only show up to 10h for clarity
        
        if i == len(sigmas) - 1:
            ax.set_xlabel("Service time (min)")
        if j == 0:
            ax.set_ylabel("Density")

plt.tight_layout()
plt.show()