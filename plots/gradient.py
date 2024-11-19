import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd 


def f_sub(x, a=1, b=2, x_3=0, c=-5, d=7, e=-6):
    return a*x**4 + b*(x-x_3)**3 + c*x**2 + d*x + e*np.exp(x)
    # return 1/(1 + np.exp(-(a*x+b)))


def f(r):
    x, y = r
    # return (1/10)*(f_sub(x, a=0, b=0, c=1, d=-20)+ f_sub(y, a=0, b=0, c=1)) 
    return (1/10)*(f_sub(x)+ f_sub(y, x_3=0, c=-2)) 






# def f(r):
#     x, y = r
    # return np.sin(x) * np.cos(y) + x**2 + y**2 - 2*x*y
    # return np.sin(x) * np.cos(y) + 0.1 * x**2 + 0.1 * y**2 + np.sin(3*x) * np.cos(3*y)
    # return np.sin(x**2 + y**2) * np.cos(2*x + 3*y) + (x**2 + y**2) / 2 - 0.5 * np.sin(5*x) * np.sin(5*y)
    # return np.sin(x**2 + y**2) * np.cos(x) + (x**2 + y**2) / 10 - np.sin(x) * np.cos(y)


eps = 0.01
lr = 0.1

def grad_dir(f, p, eps=eps, lr=lr, max_steps=20, direction=1):
    z = f(p)
    ax.scatter(*p, z, s=5, color="black")
    first_grad = nd.Gradient(f)(p)
    for _ in range(max_steps):
        z = f(p)
        grad = nd.Gradient(f)(p)

        d_grad = direction*(lr*grad)
        d = p + d_grad
        z_grad = f(d) - z

        ax.scatter(*d, f(d), s=5, color="black")
        ax.quiver(*p, z, *d_grad, z_grad, color="black", linewidth=0.5, arrow_length_ratio=0.45)
        
        plt.draw()
        plt.pause(0.1)
        if np.linalg.norm(np.array([*d, f(d)]) - np.array([*p, z])) < eps:    
            break
        p = d
    return first_grad


def minimize(f, p, eps=eps, lr=lr, max_steps=20):
    return grad_dir(f, p, eps, lr, max_steps, direction=-1)

def maximize(f, p, eps=eps, lr=lr, max_steps=20):
    return grad_dir(f, p, eps, lr, max_steps, direction=1)



# Tworzenie siatki punktów
x = np.linspace(-3.5, 3.5, 300)  # Zakres dla x
y = np.linspace(-3.5, 3.5, 300)  # Zakres dla y
X, Y = np.meshgrid(x, y)  # Siatka 2D
Z = f([X, Y])  # Obliczanie wartości funkcji na siatce


# Rysowanie wykresu 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

elevation = 30  # Kąt pionowy
azimuth = 160   # Kąt poziomy
ax.view_init(elev=elevation, azim=azimuth)

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.75)  # Powierzchnia 3D


# p = np.array([-0.5, -1]) # for minimizing
p = np.array([-1.5, -1.5]) # for maximizing
# p = np.random.uniform(low=-3.5, high=3.5, size=2)
grad = nd.Gradient(f)(p)


# title = """$f(x, y) = \\frac{1}{10}\\left(x^4 + 2x^3 - 5x^2 + 7x - 6e^x + y^4 + 2y^3 - 2y^2 + 7y - 6e^y \\right)$"""
title = """$f(x, y) = \\frac{1}{10}\\left(x^4 + 2x^3 - 5x^2 + 7x - 6e^x + y^4 + 2y^3 - 2y^2 + 7y - 6e^y \\right)$
$\\nabla f(%.1f, %.1f) = [%.3f, %.3f], \\ \\ lr = %.1f, \\ \\ motion = lr \\cdot \\nabla f(%.1f, %.1f)$""" % (*p, *grad, lr, *p)

ax.set_title(title)
# ax.text(*(p-0.1), f(p), s="$p(%.1f, %.1f, %.2f)$" % (*p, f(p)), fontsize=7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

plt.pause(1)


for i in range(1):
    # p = np.random.uniform(low=-3.5, high=3.5, size=2)
    grad = nd.Gradient(f)(p)    
    ax.scatter(*p, f(p), s=20, color="red", zorder=0)
    ax.text(*(p-0.1), f(p), s="$p(%.1f, %.1f, %.2f)$" % (*p, f(p)), fontsize=7)
    # minimize(f, p, max_steps=1)
    maximize(f, p, max_steps=1)


plt.show()