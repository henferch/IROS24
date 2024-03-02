import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.axes3d import get_test_data
import numpy as np
from numpy.linalg import inv
from Network import Network
from PlotTools import *
from Utils import Utils


fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(7, 7))
fig.tight_layout()
#plt.rcParams['axes.titley'] = 0.95   

res = 16
#res = 21
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

#print(objs_ij_refs)

dt = 0.05
ut = Utils.getInstance()
params = {\
    'ut': ut,
    'ref': sphere1.getRefs(),
    'res' : res,
    'sig': [[0.5,0.0],[0.0,0.5]],
    #'sig': [[0.75,0.0],[0.0,0.75]],
    #'h' : -0.025,
    'h_pre' : -0.01,
    'h_sel' : -0.0001,
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

# switch attention to objects
def exp1(t, ow):    
    oWeights = np.ones(len(objects))
    lWeight = 0.0
    rWeight = 0.0
    aWeight = 0.0
    bWeight = 0.0
    nWeight = 0.0    
    if (t < 1.0 ):        
        oWeights[0] = ow
    elif (t > 10.0 and t < 11.0):
        oWeights[1] = ow
    elif (t > 20.0 and t < 21.0 ):
        oWeights[2] = ow
    elif (t > 30.0 and t < 31.0 ):
        oWeights[3] = ow
    elif (t > 40.0 and t < 41.0 ):
        oWeights[4] = ow
    elif (t > 50.0 and t < 51.0 ):
        oWeights[5] = ow
    return oWeights, lWeight, rWeight, aWeight, bWeight, nWeight

# switch attention to the other side
def exp2(t, ow):    
    oWeights = np.ones(len(objects))
    lWeight = 0.0
    rWeight = 0.0
    aWeight = 0.0
    bWeight = 0.0
    nWeight = 0.0
    if (t < 1.0 ):        
        oWeights[0] = ow
    elif (t > 10.0 and t < 11.0):
        lWeight = 3.0        
    elif (t > 20.0 and t < 21.0):
        lWeight = 3.0        
    elif (t > 30.0 and t < 31.0 ):
        rWeight = 3.0
    
    return oWeights, lWeight, rWeight, aWeight, bWeight, nWeight

# switch attention to the above and below
def exp3(t, ow):
    oWeights = np.ones(len(objects))
    lWeight = 0.0
    rWeight = 0.0
    aWeight = 0.0
    bWeight = 0.0
    nWeight = 0.0
    if (t < 1.0 ):        
        oWeights[0] = ow
    elif (t > 10.0 and t < 11.0):
        aWeight = 3.0        
    elif (t > 20.0 and t < 21.0):
        bWeight = 3.0        
    # elif (t > 30.0 and t < 31.0 ):
    #     aWeight = 3.0
    
    return oWeights, lWeight, rWeight, aWeight, bWeight, nWeight

# switch to next object
def exp4(t, ow):
    oWeights = np.ones(len(objects))
    lWeight = 0.0
    rWeight = 0.0
    aWeight = 0.0
    bWeight = 0.0
    nWeight = 0.0    
    oWeights[0] = ow
    oWeights[4] = ow
    nw = 15.0    
    if (t < 1.0 ):      
        oWeights[0] = 1.0                                 
    elif (t > 10.0 and t < 11.0):
        nWeight = nw
    elif (t > 20.0 and t < 21.0):
        nWeight = nw
    elif (t > 30.0 and t < 31.0 ):
         nWeight = nw
    elif (t > 40.0 and t < 41.0 ):
         nWeight = nw
    
    return oWeights, lWeight, rWeight, aWeight, bWeight, nWeight

while (t < T):
    all_t.append(t)
    
    ow = 16.0
    # experiment to be performed
    oWeights, lWeight, rWeight, aWeight, bWeight, nWeight = exp1(t, ow)
    #oWeights, lWeight, rWeight, aWeight, bWeight, nWeight = exp2(t, ow)
    #oWeights, lWeight, rWeight, aWeight, bWeight, nWeight = exp3(t, ow)
    #oWeights, lWeight, rWeight, aWeight, bWeight, nWeight = exp4(t, ow)

    u_pre, u_sel, o = network.step({'o':oWeights, 'l' : lWeight, 'r': rWeight, 'a' : aWeight, 'b': bWeight, 'n': nWeight})
    all_u_pre.append(u_pre)
    all_u_sel.append(u_sel)
    all_o.append(o)
    t += dt
    

sphere1.draw(u_pre)
sphere2.draw(u_sel)
#sphere2.draw(network._W_pre_below)
#sphere2.draw(network._W_pre_above)
#sphere2.draw(network._W_pre_left)

fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.25, 0.14, 0.5, 0.015])

fig.colorbar(sphere1.scamap, cax=cbar_ax, location="bottom", orientation="horizontal")

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(7, 7))

ax1.set_title("Evolution of $u_\mathrm{pre}$")
u_pre = np.vstack(all_u_pre)
im1 = ax1.imshow(u_pre.transpose())
cbar_ax = fig.add_axes([0.92, 0.685, 0.015, 0.164])
fig.colorbar(im1, cax=cbar_ax, orientation='vertical')
ax1.get_xaxis().set_visible(False)

ax2.set_title("Evolution of $u_\mathrm{sel}$")
u_sel = np.vstack(all_u_sel)
im2 = ax2.imshow(u_sel.transpose())
cbar_ax = fig.add_axes([0.92, 0.4125, 0.015, 0.164])
fig.colorbar(im2, cax=cbar_ax, orientation='vertical')
ax2.get_xaxis().set_visible(False)

ax3.set_title("Object selection")
all_o = np.vstack(all_o)
for i in range(all_o.shape[1]):
    ax3.plot(np.array(all_t), all_o[:,i], label='Object {}'.format(i+1))
ax3.set_xlim([all_t[0],all_t[-1]])
ax3.legend(fontsize="7")


# plt.figure(4)
# plt.title("Selection mean")
# u_sel = np.vstack(all_u_sel)
# u_sel_mean = np.mean(u_sel.transpose(),axis=0)
# selection = 1.0 - network._ut.sigmoid(u_sel_mean, -0.8, 10.0)
# #selection = network._ut.sigmoid(u_sel_mean, -0.5, 10.0)
# #plt.plot(np.array(all_t), u_sel_mean)
# plt.plot(np.array(all_t), selection)


# plt.figure(6)
# plt.title("U Pre Matrix")
# u_sel = np.vstack(all_u_sel)
# plt.imshow(network._u_pre.reshape((res,res)))
# plt.xlim([0,res-1.0])
# plt.ylim([0,res-1.0])
# plt.colorbar()


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





