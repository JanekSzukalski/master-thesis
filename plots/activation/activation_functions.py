import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


x = np.linspace(-10, 10, 500)
fs = [
    (relu, "ReLU"),
    (sigmoid, "sigmoid"),
    (tanh, "tanh"),
    # (softmax, "")
]

for f, name in fs:
    y = f(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color="blue", linewidth=2)
    plt.xlabel("x", fontsize=12)
    plt.ylabel(f"{name}(x)", fontsize=12)
    plt.xlim(-10, 10)
    plt.ylim(y.min()-0.25, y.max()+0.25)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.savefig(f'plots/activation_{name}.png', bbox_inches='tight')

# plt.show()