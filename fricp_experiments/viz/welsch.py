import numpy as np
import matplotlib.pyplot as plt

def welsch_function(x, nu):
    return 1 - np.exp(-(x**2) / (2 * nu**2))

x = np.linspace(-1, 1, 500)
nu_values = [0.05, 0.1, 0.5]
colors = ['#d62728', '#ff7f0e', '#1f77b4'] # Rouge, Orange, Bleu

plt.figure(figsize=(8, 5))

for nu, color in zip(nu_values, colors):
    y = welsch_function(x, nu)
    plt.plot(x, y, label=r'$\nu = {}$'.format(nu), color=color, linewidth=2)

plt.title("Welsch's Function for Different Scale Parameters $\\nu$")
plt.xlabel("Residual distance $d$")
plt.ylabel(r"$\rho(d)$")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()

# Sauvegarde pour inclusion dans le LaTeX
plt.savefig("welsch_plot.pdf", format="pdf")
plt.show()