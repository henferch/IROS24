import matplotlib.pyplot as plt
from GeodesicDome import GeodesicDome
from Utils import Utils
import copy
import numpy as np


# Icosahedron
center = np.array([0.0,0.0,0.0])
tesselation = 3        
sigma = 0.005
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
# right 
N = egoSphere1.v_N
nObjects = len(objs_int) 

W_pre_right = np.zeros((N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        
W_pre_left = np.zeros((N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        
W_pre_above = np.zeros((N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        
W_pre_below = np.zeros((N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        

oW = np.zeros((nObjects,), dtype=np.dtype)
ref = egoSphere1.getV()
right = 1.0
left = 0.0
above = 1.0
below = 1.0
oW[1] = 1.0

# precomputed const 
sigmaLR = np.eye(3)*0.5
sigmaLR[2,2] = 0.008
invSigmaLR, denLR = ut.getMultiGaussianConst(sigmaLR)

sigmaAB = np.eye(3)*0.5
sigmaAB[1,1] = 0.008
invSigmaAB, denAB = ut.getMultiGaussianConst(sigmaAB)

offset = 0.0
if right > 0.0:
    MU = 0.0
    for i in range(nObjects):
        MU += oW[i]*objs_int[i]
    W_pre_right = ut.sigmoid(ref[:,1] - MU[1] - offset,0, 50.0) * ut.multiGaussianPrecomp(MU, ref, invSigmaLR, denLR)                

if left > 0.0:
    MU = 0.0
    for i in range(nObjects):
        MU += oW[i]*objs_int[i]
    W_pre_right = ut.sigmoid(MU[1] - ref[:,1] + offset,0, 50.0) * ut.multiGaussianPrecomp(MU, ref, invSigmaLR, denLR)                

if above > 0.0:
    MU = 0.0
    for i in range(nObjects):
        MU += oW[i]*objs_int[i]
    W_pre_right = ut.sigmoid(ref[:,2] - MU[2] + offset ,0, 50.0) * ut.multiGaussianPrecomp(MU, ref, invSigmaAB, denAB)                

if below > 0.0:
    MU = 0.0
    for i in range(nObjects):
        MU += oW[i]*objs_int[i]
    W_pre_right = ut.sigmoid(MU[2] - ref[:,2] - offset,0, 50.0) * ut.multiGaussianPrecomp(MU, ref, invSigmaAB, denAB)

# if left > 0.0:
#     MU = 0.0
#     for i in range(nObjects):
#         MU += oW[i]*objs_int[i]                
#     W_pre_right = ut.sigmoid(ref[:,1] - MU[1],0, 50.0) * ut.gaussian(MU[2], ref[:,2], 0.08)        
    
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(7, 7))
fig.tight_layout()

for o in objs_int:
    ut.plot3DPoint(ax, o, 'red')

egoSphere1.plot(ax, W_pre_right)
ax.view_init(elev=0, azim=0)
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])
ax.grid(False)
ax.axis('off')
ax.set_aspect('equal')
plt.show()
