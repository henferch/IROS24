import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.axes3d import get_test_data
import numpy as np
from numpy.linalg import inv
from NetworkGeodesic import NetworkGeodesic
#from PlotTools import *
from GeodesicDome import GeodesicDome
from Utils import Utils
import copy

fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(7, 7))
fig.tight_layout()

# Icosahedron
center = np.array([0.0,0.0,0.0])
tesselation = 3        
sigma = 0.005
sigma = 0.01
radius = 1.0
objects = [np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0])]
params = {'tesselation': tesselation, 'scale' : radius, 'center': center}        
egoSphere1 = GeodesicDome(params)
egoSphere2 = copy.copy(egoSphere1)

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
    _, p = egoSphere1.intersect(center, center-o)
    objs_int.append(p)

ut = Utils.getInstance()
refs = egoSphere1.getV()
dt = 0.05        
params = {\
    'ut': ut,    'h_pre' : -0.01,
    'ref': refs,
    'sig': np.eye(3)*sigma,
    'h_pre' : -0.01,
    #'h_sel' : -0.0001,
    'h_sel' : -0.0001,
    'dt' : dt,
    #'inh' : 0.0001,
    'inh' : 0.0001,
    'tau_pre' : 0.2,
    'tau_sel' : 0.2,
    'o_alpha': 250.0,
    'objects': objs_int
    }

network = NetworkGeodesic(params) 
u_pre = None
u_mem = None
t = 0.0
T = 60.0
all_u_pre = []
all_u_sel = []
all_o = []
all_t = []

# switch attention to objects
def exp1(t, o_base, o_act):    
    oWeights = np.ones(len(objects))*o_base
    lWeight = 0.0
    rWeight = 0.0
    aWeight = 0.0
    bWeight = 0.0
    nWeight = 0.0    
    if (t < 1.0 ):        
        oWeights[0] = o_act
    elif (t > 10.0 and t < 11.0):
        oWeights[1] = o_act
    elif (t > 20.0 and t < 21.0 ):
        oWeights[2] = o_act
    elif (t > 30.0 and t < 31.0 ):
        oWeights[3] = o_act
    elif (t > 40.0 and t < 41.0 ):
        oWeights[4] = o_act
    elif (t > 50.0 and t < 51.0 ):
        oWeights[5] = o_act
    return oWeights, lWeight, rWeight, aWeight, bWeight, nWeight

# switch attention to the other side
def exp2(t, o_base, o_act):    
    oWeights = np.ones(len(objects))*o_base
    lWeight = 0.0
    rWeight = 0.0
    aWeight = 0.0
    bWeight = 0.0
    nWeight = 0.0
    if (t < 1.0 ):        
        oWeights[0] = o_act
    elif (t > 10.0 and t < 11.0):
        lWeight = o_act*3.0        
    elif (t > 20.0 and t < 21.0):
        lWeight = o_act*3.0        
    elif (t > 30.0 and t < 31.0 ):
        rWeight = o_act*3.0
    
    return oWeights, lWeight, rWeight, aWeight, bWeight, nWeight

# switch attention to the above and below
def exp3(t, o_base, o_act):
    oWeights = np.ones(len(objects))*o_base
    lWeight = 0.0
    rWeight = 0.0
    aWeight = 0.0
    bWeight = 0.0
    nWeight = 0.0
    if (t < 1.0 ):        
        oWeights[0] = o_act
    elif (t > 10.0 and t < 11.0):
        aWeight = o_act*3.0  
    elif (t > 20.0 and t < 21.0):
        bWeight = o_act*3.0  
    # elif (t > 30.0 and t < 31.0 ):
    #     aWeight = 3.0
    
    return oWeights, lWeight, rWeight, aWeight, bWeight, nWeight

# switch to next object
def exp4(t, o_base, o_act):
    oWeights = np.ones(len(objects))*o_base
    lWeight = 0.0
    rWeight = 0.0
    aWeight = 0.0
    bWeight = 0.0
    nWeight = 0.0    
    oWeights[0] = o_act
    #oWeights[2] = o_act
    oWeights[4] = o_act
    if (t < 1.0 ):      
        oWeights[0] = o_base
        #oWeights[2] = o_base               
    elif (t > 10.0 and t < 10.5):
        nWeight = 0.75
    elif (t > 20.0 and t < 20.5):
        nWeight = 0.75
    elif (t > 30.0 and t < 30.5 ):
         nWeight = 0.75
    elif (t > 40.0 and t < 40.5 ):
         nWeight = 0.75
    
    return oWeights, lWeight, rWeight, aWeight, bWeight, nWeight

while (t < T):
    all_t.append(t)
    o_base = 0.9
    o_act = 12.0
    o_act = 9.0
    
    # experiment to be performed
    #oWeights, lWeight, rWeight, aWeight, bWeight, nWeight = exp1(t, o_base, o_act)
    oWeights, lWeight, rWeight, aWeight, bWeight, nWeight = exp2(t, o_base, o_act)
    #oWeights, lWeight, rWeight, aWeight, bWeight, nWeight = exp3(t, o_base, o_act)
    #oWeights, lWeight, rWeight, aWeight, bWeight, nWeight = exp4(t, o_base, o_act)

    u_pre, u_sel, o = network.step({'o':oWeights, 'l' : lWeight, 'r': rWeight, 'a' : aWeight, 'b': bWeight, 'n': nWeight})
    all_u_pre.append(u_pre)
    all_u_sel.append(u_sel)
    all_o.append(o)
    t += dt
    
egoSphere1.plot(ax[0], u_pre)
# for o in objs_int:
#     ut.plot3DPoint(ax[0], o, 'red')
for o,p in zip(objects,objs_int):
    ut.plot3DLine(ax[0], o, p, 'red', 0.4, 'dashdot')
    ut.plot3DPoint(ax[0], o, 'black')
egoSphere2.plot(ax[1], u_sel)
for a in ax:
    a.view_init(elev=0, azim=0)
    a.set_xlim([-1.5,1.5])
    a.set_ylim([-1.5,1.5])
    a.set_zlim([-1.5,1.5])
    a.grid(False)
    a.axis('off')
    a.set_aspect('equal')

#egoSphere2.plot(ax[1], u_sel)
#sphere2.draw(network._W_pre_below)
#sphere2.draw(network._W_pre_above)
#sphere2.draw(network._W_pre_left)

fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.25, 0.14, 0.5, 0.015])

fig.colorbar(egoSphere1.scamap, cax=cbar_ax, location="bottom", orientation="horizontal")

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





