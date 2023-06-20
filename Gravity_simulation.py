import numpy as np
import matplotlib.pyplot as plt
import imageio

# constants

# Added a comment

G = 6.67e-11

m1 = 1e11
m2 = 1e11

p1 = np.array([-2000, -3000])
p2 = np.array([2000, 3000]) 

v1 = np.array([0.01, 0]) 
v2 = np.array([-0.01, 0])

def force(r1, r2):
    
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    
    d = np.sqrt(dx**2 + dy**2)
    
    F = -G*m1*m2/d**2
    

    Fx = F * dx/d
    Fy = F * dy/d
    
    return np.array([Fx, Fy])

for i in range(1000000):
    
    dt = i/10000
    
    v1 = v1 + dt*force(p1, p2)/m1
    v2 = v2 + dt*force(p2, p1)/m2
    
    p1 = p1 + v1*dt
    p2 = p2 + v2*dt
    #print(force(p1, p2), v1, v2)
    if i % 1000 == 0:
        plt.scatter(p1[0], p1[1], label="p1")
        plt.scatter(p2[0], p2[1], label="p2")
        plt.legend()
        plt.ylim(-10000, 10000)
        plt.xlim(-10000, 10000)
        plt.title("frame {}".format(i/1000))
        plt.savefig("Frames/Test_{}.jpeg".format(int(i/1000)),bbox_inches='tight',dpi=200)
        plt.close("all")
        print(i/1000)
    
    if abs(p1[0]) > 10000 or abs(p1[1]) > 10000:
        break

    
    
with imageio.get_writer('gravity.gif', mode='I',duration=0.07) as writer:
    for i in range(1000):
        print(i)
        image = imageio.imread("Test_{}.jpeg".format(i))
        writer.append_data(image)
    
#Cringe increased again
    
