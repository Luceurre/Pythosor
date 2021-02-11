import numpy as np
import matplotlib.pyplot as plt
from tensor import Tensor
import pickle

# Setup

Nx = 128
Nv = 128

T = 10
dt = 0.001

beta = 1e-2 #10e-2
delta = 0.2
v0 = 5  #2.4

def f0x(x):
    return (1 + beta * np.cos(delta * x))

def f0v(v):
    return (1 / (2 * np.sqrt(2 * np.pi))) * (np.exp(-0.5 * (v - v0)**2) + np.exp(-0.5 * (v + v0)**2))

xmin = 0.
xmax = 10.
vmin = -10.
vmax = 10.

gridx = np.linspace(xmin, xmax, Nx)
gridv = np.linspace(vmin, vmax, Nv)

dx = (xmax - xmin) / Nx
dv = (vmax - vmin) / Nv

f0 = Tensor.from_functions([f0x, f0v], [gridx, gridv])
# f0 = Tensor.rand((Nx, Nv), 1)

# Equation

gradx = np.zeros((Nx, Nx))
gradv = np.zeros((Nv, Nv))

for i in range(Nx):
    gradx[i, (i + 1) %  Nx] = 1
    gradx[i, i - 1] = -1
for i in range(Nv):
    gradv[i, (i + 1) % Nv] = 1
    gradv[i, i - 1] = -1

gradx /= (2. * dx)
gradv /= (2. * dv)

def E(f):
    return f.integrate(1, lambda fx: dx * fx).untensorized()

def rho(f):
    return np.diag(1. - E(f))

v = np.diag(gridv)

identity = Tensor.eye((Nx, Nv))

# Simulation

solution = [f0]
n = 0

while n * dt < T:
    current_f = solution[n]
    operator_1 = Tensor.from_arrays([np.eye(Nx), v.dot(gradv)])
    operator_2 = Tensor.from_arrays([rho(current_f).dot(gradx), np.eye(Nv)])

    full_operator = identity + dt * (operator_1 + operator_2)

    next_f = full_operator * current_f
    print(f"Rank at t = {n * dt} before compression: {next_f.rank} with norm: {next_f.norm()}")
    next_f_compressed = next_f.compress()

    solution.append(next_f_compressed)
    n += 1

solution[0].draw()
input()
solution[-1].draw()
input()

pickle.dump(solution, open("data.pk", "wb"))
