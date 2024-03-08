import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.axes3d import get_test_data
import numpy as np
from numpy.linalg import inv
from GeodesicDome import GeodesicDome 
from Utils import Utils
import copy


fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(7, 7))
fig.tight_layout()

# Icosahedron
center = np.array([0.0,0.0,0.0])
tesselation = 3        
sigma = 0.005
sigma = 0.01
radius = 1.0
params = {'tesselation': tesselation, 'scale' : radius, 'center': center}        
egoSphere = GeodesicDome(params)
ut = Utils.getInstance()

objects = []

for i in range(2):
    p = np.array([1.5,0.5,-0.2 + i *(0.4)])
    objects.append(p)

for i in range(2):
    p = np.array([1.5,0.0,-0.2 + i *(0.4)])
    objects.append(p)

for i in range(2):
    p = np.array([1.5,-0.5,-0.2 + i *(0.4)])
    objects.append(p)

objs_int = []
for o in objects:
    _, p = egoSphere.intersect(center, center-o)
    objs_int.append(p)

state = np.zeros((egoSphere.v_N,), dtype=np.float32)
state[0] = 0.001
state[5] = -0.001
egoSphere.cmap = 'GnBu'
egoSphere.plot(ax, state, 0.4, False, True)

print(objects, objs_int)
k = 1
for o,p in zip(objects,objs_int):
    ut.plot3DLine(ax, o, p, 'darkcyan', 3.0, 'dashed')
    ut.plot3DPoint(ax, o, 'orangered', 's', 7.0)
    ut.plot3DPoint(ax, p, 'orangered', 'x', 7.0)
    #ut.plot3DPoint(ax, o, 'orangered', '${}$'.format(k), 10.0)
    k += 1

ax.view_init(elev=0, azim=0)
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])
ax.grid(False)
ax.axis('off')
ax.set_aspect('equal')


plt.show()
