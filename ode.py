from tensor import Tensor
import numpy as np
import matplotlib.pyplot as plt

def f0x(x):
    return x

def f0v(v):
    return 1.

Nx = 128
Nv = 128
dt = 0.005
T = 10
time = np.arange(0., T, dt)

gridx = np.linspace(0., 10., Nx)
gridv = np.linspace(0., 10., Nv)

f0 = Tensor.from_functions([f0x, f0v], [gridx, gridv])

df = Tensor.from_arrays([np.eye(Nx), np.eye(Nv)])
identity = Tensor.eye([Nx, Nv])

operator = identity + dt * df
point_of_interest = []

for i in time:
    f0 = operator * f0
    f0 = f0.compress()
    point_of_interest.append(f0((Nx - 1, 0)))

p, a = plt.subplots()
a.plot(time, point_of_interest)
p.show()
input()
