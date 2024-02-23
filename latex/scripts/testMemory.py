import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.axes3d import get_test_data
import numpy as np
from numpy.linalg import inv
from Network import Network
from PlotTools import *


fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(7, 7))
fig.tight_layout()
plt.rcParams['axes.titley'] = 0.95   

res = 17
sphere1 = Sphere(ax[0], res, 'Sensory Ego-Sphere', 'viridis')
sphere2 = Sphere(ax[1], res, 'Working Memory', 'viridis')

objects = []
zero = np.array([0.0,0.0,0.0])

for i in range(2):
    p = np.array([-0.4,0.5,-0.3 + i *(0.3)])
    objects.append(p)
objs_ij_refs = []
for o in objects:
    p, _ = sphere1.intersect(zero, o - zero)
    i,j = sphere1.getij(p)
    objs_ij_refs.append(np.array([i,j]))
    sphere1.plot3DPoint(sphere1.ax, p, 'red')
    #sphere1.plot3DLine(sphere1.ax, p, o, 'red')

dt = 0.05
params = {\
    'ref': sphere1.getRefs(),
    'sig': [[1.0,0.0],[0.0,1.0]],
    'h' : -0.025,
    'dt' : dt,
    'inh' : 0.001,
    'tau' : 2.0,
    }              

network = Network(params) 
u = None
u_mem = None
t = 0.0
T = 15.0
all_u = []
all_u_mem = []
while (t < T):
    if (t > 1.0 and t < 3.2 ):
        u, u_mem = network.step([objs_ij_refs[0]])
    else:
        u, u_mem = network.step(objs_ij_refs)
    all_u.append(u)
    all_u_mem.append(u_mem)
    t += dt

sphere1.draw(u)
sphere2.draw(u_mem)

fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.25, 0.14, 0.5, 0.015])

fig.colorbar(sphere1.scamap, cax=cbar_ax, location="bottom", orientation="horizontal")
plt.show()

