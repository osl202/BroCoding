import numpy as np
import matplotlib.pyplot as plt

nu = 0.01/np.pi

nx = 501


dx = 2*np.pi/nx
dt = nu*dx

nt = int(1/dt)

x = np.linspace(0, 2*np.pi, nx)
t = np.linspace(0, dt*nt, nt)
u = np.empty(nx)

u = np.sin(x)
un = np.zeros((nx, nt))
un[:,0] = u

#plt.plot(x, u, label="init")

for i in range(nt):
    for j in range(1, nx - 1):
        u[j] = u[j] - u[j]*(dt/dx)*(u[j] - u[j-1]) + nu*(dt/dx**2)*(u[j+1] - 2*u[j] + u[j-1])
        u[0] = u[0] - u[0]*(dt/dx)*(u[0] - u[-2]) + nu*(dt/dx**2)*(u[1] - 2*u[0] + u[-2])
        u[-1] = u[0]
    #print(len(un[:,0]), len(un[0,:]))
    un[:,i] = u
    
T, X = np.meshgrid(t, x)
print(len(t))
plt.contourf(T, X,un)
#plt.plot(x, u, label="final")
plt.show()