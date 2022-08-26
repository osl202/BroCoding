import numpy
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

nx = 41
ny = 41
nt = 120
c = 1
dx = 1#2 / (nx - 1)
dy = 1#2 / (ny - 1)
sigma = .009
nu = 1
dt = 0.1


x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

X, Y = numpy.meshgrid(x, y)

u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
T = numpy.zeros((ny, nx))

u[0, :] = 1
u[:, 0] = 0
u[-1, :] = 0
u[:, -1] = 0

T[0, :] = 100
T[:, 0] = 0
T[-1, :] = 0
T[:, -1] = 0


#u[1:-1, 0:-2] = 3
#u[1:-1, 2:] = 2
#u[2:,1: -1] = 4
#u[0:-2, 1:-1] = 5

v[0, :] = 0
v[:, 0] = 0
v[-1, :] = 100
v[:, -1] = 0



for i in range(10000):
    un = u.copy()
    vn = v.copy()
    
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * 
                     (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - 
                     dt / dy * vn[1:-1, 1:-1] * 
                     (un[1:-1, 1:-1] - un[0:-2, 1:-1]) + 
                     nu * dt / dx**2 * 
                     (un[1:-1,2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + 
                     nu * dt / dy**2 * 
                     (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - 
                     dt / dx * un[1:-1, 1:-1] *
                     (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     dt / dy * vn[1:-1, 1:-1] * 
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) + 
                     nu * dt / dx**2 * 
                     (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     nu * dt / dy**2 *
                     (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
    
    T[1:-1,1:-1] = (T[1:-1,1:-1] + 
                    
                    dt*(
                    (T[1:-1,2:] - 2 * T[1:-1,1:-1] + T[1:-1,0:-2])/(dx**2) +
                    (T[2:, 1:-1] - 2 * T[1:-1,1:-1] + T[0:-2, 1:-1])/(dy**2) -
                     
                    un[1:-1, 1:-1] *
                    (T[1:-1, 1:-1] - T[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy *
                    (T[1:-1, 1:-1] - T[0:-2, 1:-1])))
                    
                    
                    
    
    
    u[0, :] = 1
    u[:, 0] = 0
    u[-1, :] = 0
    u[:, -1] = 0
    v[0, :] = 0
    v[:, 0] = 0
    v[-1, :] = 0
    v[:, -1] = 0
    T[0, :] = 0
    T[:, 0] = 0
    T[-1, :] = 0
    T[:, -1] = 0
    

    #print(u)
plt.contourf(X, Y, T, alpha=0.5, cmap=cm.viridis) 
plt.show()