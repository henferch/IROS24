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

res = 16
sphere1 = Sphere(ax[0], res, 'Pre-selection', 'viridis')
sphere2 = Sphere(ax[1], res, 'Selection', 'viridis')

objects = []
zero = np.array([0.0,0.0,0.0])

for i in range(2):
    p = np.array([-0.4,0.5,-0.2 + i *(0.4)])
    objects.append(p)

for i in range(2):
    p = np.array([0.0,0.5,-0.2 + i *(0.4)])
    objects.append(p)

for i in range(2):
    p = np.array([0.4,0.5,-0.2 + i *(0.4)])
    objects.append(p)

# p = np.array([0.4,0.5,0.3])
# objects.append(p)

objs_ij_refs = []
for o in objects:
    p, _ = sphere1.intersect(zero, o)
    i,j = sphere1.getij(p)
    objs_ij_refs.append(np.array([i,j]))
    # sphere1.plot3DPoint(sphere1.ax, p, 'red')
    # sphere1.plot3DLine(sphere1.ax, zero, o, 'red')
    # sphere1.plot3DPoint(sphere1.ax, o, 'black')

print(objs_ij_refs)

dt = 0.05
params = {\
    'ref': sphere1.getRefs(),
    'res' : res,
    'sig': [[0.5,0.0],[0.0,0.5]],
    #'sig': [[0.75,0.0],[0.0,0.75]],
    'h' : -0.025,
    'dt' : dt,
    'inh' : 0.001,
    'tau' : 2.0,
    'objects': objs_ij_refs
    }              

network = Network(params) 
u_pre = None
u_mem = None
t = 0.0
T = 60.0
all_u_pre = []
all_u_sel = []
all_o = []
all_t = []

while (t < T):
    all_t.append(t)
    oWeights = np.ones(len(objects))
    lWeight = 0.0
    rWeight = 0.0
    nWeight = 0.0

    # if (t < 3.0 ):
    #     oWeights[0] = 4.0
    # elif (t > 10.0 and t < 10.1):
    #     #lWeight = 2.0
    #     nWeight = 1.5

    if (t < 3.0 ):
        #network._u_sel -= 10.5
        #u_pre, u_sel = network.step(objs_ij_refs)
        oWeights[0] = 4.0
    elif (t > 15.0 and t < 18.0):
        oWeights[1] = 4.0        
    elif (t > 25.0 and t < 28.1 ):
        oWeights[2] = 4.0
    elif (t > 35.0 and t < 38.1 ):
        oWeights[0] = 4.0
    elif (t > 40.0 and t < 42.1 ):
        lWeight = 2.0 
    elif (t > 45.0 and t < 47.1 ):
        rWeight = 2.0
    elif (t > 80.0 and t < 82.1 ):
        lWeight = 2.0
    
    # else:
    #     oWeights = np.ones(len(objects))
    u_pre, u_sel, o = network.step({'o':oWeights, 'l' : lWeight, 'r': rWeight, 'n': nWeight})
    all_u_pre.append(u_pre)
    all_u_sel.append(u_sel)
    all_o.append(o)
    t += dt
    
sphere1.draw(u_pre)
sphere2.draw(u_sel)

fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.25, 0.14, 0.5, 0.015])

fig.colorbar(sphere1.scamap, cax=cbar_ax, location="bottom", orientation="horizontal")

plt.figure(2)
plt.title("Evolution of U")
u_pre = np.vstack(all_u_pre)
plt.imshow(u_pre.transpose())
plt.colorbar()

plt.figure(3)
plt.title("Evolution of U sel")
u_sel = np.vstack(all_u_sel)
plt.imshow(u_sel.transpose())
plt.colorbar()

# plt.figure(4)
# plt.title("Selection mean")
# u_sel = np.vstack(all_u_sel)
# u_sel_mean = np.mean(u_sel.transpose(),axis=0)
# selection = 1.0 - network._ut.sigmoid(u_sel_mean, -0.8, 10.0)
# #selection = network._ut.sigmoid(u_sel_mean, -0.5, 10.0)
# #plt.plot(np.array(all_t), u_sel_mean)
# plt.plot(np.array(all_t), selection)

plt.figure(5)
plt.title("Object selection")
all_o = np.vstack(all_o)
for i in range(all_o.shape[1]):
    #plt.plot(np.array(all_t), selection*all_o[:,i])
    plt.plot(np.array(all_t), all_o[:,i])

plt.figure(6)
plt.title("U Pre Matrix")
u_sel = np.vstack(all_u_sel)
plt.imshow(network._u_pre.reshape((res,res)))
plt.xlim([0,res-1.0])
plt.ylim([0,res-1.0])
plt.colorbar()


# plt.figure(6)
# plt.title("left W")
# u_sel = np.vstack(all_u_sel)
# plt.imshow(network._W_pre_left.reshape((res,res)))
# plt.colorbar()

# plt.figure(7)
# plt.title("right W")
# u_sel = np.vstack(all_u_sel)
# plt.imshow(network._W_pre_right.reshape((res,res)))
# plt.colorbar()


plt.show()





