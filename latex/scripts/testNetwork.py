import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.axes3d import get_test_data
import numpy as np
from numpy.linalg import inv
from Network import Network
from PlotTools import *

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(7, 7))
fig.tight_layout()
plt.rcParams['axes.titley'] = 0.95   

#fig, ax = plt.subplots()

res = 17
sphere1 = Sphere(ax, res, 'Point excitation', 'viridis')
gauss = Gaussian(5.0)

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
    #sphere1.plot3DPoint(sphere1.ax, p, 'red')
    #sphere1.plot3DLine(sphere1.ax, zero, o, 'black')

dt = 0.05
params = {\
    'ref': sphere1.getRefs(),
    'sig': [[1.0,0.0],[0.0,1.0]],
    'h' : -0.005,
    #'h' : -0.0008867,
    'dt' : dt,
    'inh' : 0.0,
    'tau' : 1.0,
    'objects': objs_ij_refs
    }              

network = Network(params) 
u = None
u_mem = None
t = 0.0
T = 5.0
all_u_pre = []
all_u_mem = []
oWeights = np.ones(len(objects))
while (t < T):
    u_pre, u_mem, obj = network.step({'o':oWeights})
    all_u_pre.append(u_pre)
    all_u_mem.append(u_mem)
    t += dt


# all_u = np.vstack(all_u)
# all_u_mem = np.vstack(all_u_mem)

# print(len(all_u))
# print(all_u[0].shape)
# print(all_u[-1])
#plt.imshow(network._W_mem)
#plt.imshow(sphere1.getRefs())
#plt.imshow(all_u.transpose())
#plt.imshow(all_u_mem.transpose())
#print(sphere1.getRefs())
# plt.colorbar()
# plt.show()

# h = -0.01
# t = 0.0
# T = 50.0
# all_u_1 = []
# all_u_2 = []
# all_t = []
# # u_1 = np.random.normal(0.0, 0.01, 1)[0]
# # u_2 = np.random.normal(0.0, 0.01, 1)[0]
# u_1 = 0.0
# u_2 = 0.0
# w = -1.0
# dt = 0.05
# tau = 2.0

# while (t < T):
#     all_t.append(t)
#     du_1 = -u_1 + h + w*(1.0 / (1.0 + math.exp(-50*u_2))) + np.random.normal(0.0, 0.01, 1)[0] 
#     du_2 = -u_2 + h + w*(1.0 / (1.0 + math.exp(-50*u_1))) + np.random.normal(0.0, 0.01, 1)[0]
#     if t > 1.0 and t < 2.0 :
#         du_2 += 1.0

#     if t > 5.0 and t < 15.0 :
#         du_1 += 1 
#     u_1 += dt*du_1/tau 
#     u_2 += dt*du_2/tau 
#     all_u_1.append(u_1)
#     all_u_2.append(u_2)
#     t += dt

# all_t = np.array(all_t)
# all_u_1 = np.array(all_u_1)
# all_u_2 = np.array(all_u_2)
# plt.plot(all_t, all_u_1)
# plt.plot(all_t, all_u_2)
# plt.show()

sphere1.draw(u_mem)

fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.25, 0.14, 0.5, 0.015])

fig.colorbar(sphere1.scamap, cax=cbar_ax, location="bottom", orientation="horizontal")
plt.show()

