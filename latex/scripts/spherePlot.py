import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.axes3d import get_test_data
import numpy as np
from numpy.linalg import inv
import math
from Network import Network

from PlotTools import *
# class Gaussian:
#     def __init__(self, sigma):
#         sig = sigma
#         SIGMA = np.array([[1.0,0.0],[0.0,1.0]])*sig
#         self.inv_SIGMA = inv(SIGMA)
#         self.denom = np.sqrt( ((2.0*np.pi)**2.0) * np.linalg.det(SIGMA) )
#     def compute(self, K, MU):
#         K_MU = K-MU
#         return np.exp(-0.5* np.matmul(np.matmul(K_MU, self.inv_SIGMA), K_MU.transpose())) / self.denom            
# class Sigmoid:
#     def __init__(self, alpha):
#         self.alpha = alpha
#     def compute(self, k, mu):
#         return 1.0/(1.0 + math.exp(self.alpha*(k-mu)))

# class Network:
#     def __init__(self, h, tau, eps, res, esphere):
#         self.res = res
#         self.N = res*res
#         self.eps = eps
#         self.tau = tau
#         self.esphere = esphere
#         # self.tau = 1.0/tau
#         # self.o_tau = 1.0 - self.tau
#         self.u = np.ones((self.N))*h
#         self.u_mem = np.ones((self.N))*h
#         self.gauss = Gaussian(1.0)
#         self.W = np.zeros((self.N,self.N))
#         self.Wmem = np.zeros((self.N,self.N))
#         self.alphaMem = -1.0
#         k = 0
#         self.MU = []
#         for i in range(self.res):
#             for j in range(self.res):
#                 self.MU.append(np.array([i,j]))
#         self.MU = np.vstack(self.MU)
#         print(self.MU.shape)
#         k = 0
#         for i in range(self.res):
#             for j in range(self.res):
#                 i_j = np.array([i,j])
#                 self.W[k,:] = self.gauss.compute(i_j, self.MU)
#                 w_mem = np.ones((self.N))*self.alphaMem
#                 w_mem[k] = 0.0
#                 self.Wmem[k,:] = w_mem
#                 k += 1
                
#     # words = ["next", "another", "opposite"]    
#     def step(self, dt, objs, focus, words, arm):
#         syn = np.zeros((self.N,))
#         sti = np.zeros((self.N,))
#         wor = np.zeros((self.N,))
#         k = 0
#         syn_mem = np.zeros((self.res,self.res))
#         for i in range(self.res):
#             for j in range(self.res):
#                 #i_j = self.esphere.getXYZ(i,j)
#                 i_j = np.array([i,j])
#                 # synapse
#                 syn[k] = np.dot(self.W[k],self.u) + self.eps
#                 syn_mem[k] = np.dot(self.Wmem[k],self.u_mem)
#                 # stimulation
#                 st_ko = 0.0
#                 for o in objs:
#                     oo = self.esphere.getij(o)
#                     st_ko += self.gauss.compute(i_j, oo)
#                 ff = self.esphere.getij(focus) 
#                 sti[k] = st_ko + self.gauss.compute(i_j, ff)
#                 k += 1
#         # stimulation
#         self.u_mem += dt*self.tau*(-self.u_mem + self.h + syn_mem)    
#         self.u += dt*self.tau*(-self.u + self.h + syn + sti)   
#         return self.u, self.u_mem 
    
fig, ax = plt.subplots(3,2, subplot_kw={'projection': '3d'}, figsize=(7, 7))
fig.tight_layout()
plt.rcParams['axes.titley'] = 0.95   

gauss = Gaussian(5.0)
sigmoid = Sigmoid(-1.0)

sphere1 = Sphere(ax[0,0], 37, 'Point excitation', 'viridis')
sphere2 = Sphere(ax[0,1], 37, 'Point inhibition', 'viridis')
sphere3 = Sphere(ax[1,0], 37, 'Left inhibition', 'viridis')
sphere4 = Sphere(ax[1,1], 37, 'Right inhibition', 'viridis')
sphere5 = Sphere(ax[2,0], 37, 'Network', 'viridis')
sphere6 = Sphere(ax[2,1], 37, 'Network memory', 'viridis')

# sphere 1
#C = sphere1.C
MU = np.array([sphere1.resol/2.0,sphere1.resol/2.0])
g = []
for i in range(sphere1.resol):
    for j in range(sphere1.resol):
        K = np.array([i,j])
        g.append(gauss.compute(K, MU))
        #C[i,j] = gauss.compute(K, MU)
g = np.array(g)
sphere1.draw(g)

# sphere 2
sphere2.draw(1.0-g)

# sphere 3
sigmoid = Sigmoid(-1.0)
mu =  sphere1.resol/2.0 - (sphere1.resol/16.0)
s = []
for i in range(sphere1.resol):
    for j in range(sphere1.resol):
        s.append(sigmoid.compute(j, mu))

s = np.array(s) 
sphere3.draw(s)

# sphere 4
s = []
mu =  sphere1.resol/2.0 + (sphere1.resol/16.0)
for i in range(sphere1.resol):
    for j in range(sphere1.resol):
        s.append(sigmoid.compute(mu, j))
s = np.array(s) 
sphere4.draw(s)

# sphere 5
# simulated objects
objects = []
for i in range(3):
    objects.append(np.array([-0.5,0.5,-0.3 + i *(0.1)]))
for i in range(3):
    objects.append(np.array([-0.7,0.5,-0.3 + i *(0.1)]))    
for i in range(3):
    objects.append(np.array([0.5,0.5,-0.3 + i *(0.1)]))
for i in range(3):
    objects.append(np.array([0.7,0.5,-0.3 + i *(0.1)]))    
for p in objects:
    sphere5.plot3DPoint(ax[2,0], p, 'black')
sphere5.draw(np.zeros((sphere1.resol*sphere1.resol,)))    

# sphere 6
objs_ij_refs = []
for o in objects:
    i,j = sphere6.getij(o)
    objs_ij_refs.append(np.array([i,j]))
    #sphere6.plot3DPoint(sphere6.ax, sphere6.getXYZ(i,j), 'red')

dt = 0.05
params = {\
    'ref': sphere6.getRefs(),
    'sig': [[1.0,0.0],[0.0,1.0]],
    'h' : -1.1,
    'dt' : dt,
    'inh' : 0.01,
    'tau' : 5.0,
    }              

# network = Network(params) 
# u = None
# umem = None
# t = 0.0
# T = 5.0
# while (t < T):
#     u, umem = network.step(objs_ij_refs)
#     t += dt

# k = 0
# sphere6.draw(u)    


fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.25, 0.14, 0.5, 0.015])


fig.colorbar(sphere1.scamap, cax=cbar_ax, location="bottom", orientation="horizontal")
plt.show()


# t = 0.0
# T = 2.0
# dt = 0.01
# net = Network(h=-0.1, tau=4.0, eps=0.001, res=37, esphere=sphere5)
# u = net.u
# u_mem = net.u_mem
# zero = np.array([0.0,0.0,0.0])
# # obj_proj = []
# # for o in objects:
# #     obj_proj.append(sphere5.intersect(zero, o))
# while (t < T):
#     u,u_mem = net.step(dt, objects, objects[0], [],[])


# C6 = sphere6.C
# # k = 0
# # for i in range(sphere1.resol):
# #     for j in range(sphere1.resol):
# #         C5[i,j] = u[k]
# #         C6[i,j] = u_mem[k]
# #         k += 1

# sphere6.draw(C6)



# sphere1 = Sphere(ax[0,0], 37, 'Point excitation', 'viridis')
# sphere2 = Sphere(ax[0,1], 37, 'Point inhibition', 'viridis')
# sphere3 = Sphere(ax[1,0], 37, 'Left inhibition', 'viridis')
# sphere4 = Sphere(ax[1,1], 37, 'Right inhibition', 'viridis')

# C = sphere1.C
# MU = np.array([sphere1.resol/2.0,sphere1.resol/2.0])

# for i in range(sphere1.resol):
#     for j in range(sphere1.resol):
#         K = np.array([i,j])
#         C[i,j] = gauss.compute(K, MU)

# sphere1.draw(C)
# sphere2.draw(1.0-C)
# C = sphere3.C
# mu =  sphere1.resol/2.0 - (sphere1.resol/8.0)
# for i in range(sphere1.resol):
#     for j in range(sphere1.resol):
#         C[i,j] = sigmoid.compute(j, mu)
# sphere3.draw(C)

# C = sphere4.C
# mu =  sphere1.resol/2.0 + (sphere1.resol/8.0)
# for i in range(sphere1.resol):
#     for j in range(sphere1.resol):
#         C[i,j] = sigmoid.compute(mu, j)
# sphere4.draw(C)

# sphere5 = Sphere(ax[2,0], 37, 'Network', 'viridis')
# sphere6 = Sphere(ax[2,1], 37, 'Network memory', 'viridis')

# sphere6.draw(C)
# # p = sphere6.getXYZ(int(37/2),int(37/2))
# # plot3DPoint(sphere6.ax, p, 'black')

# t = 0.0
# T = 2.0
# dt = 0.01
# net = Network(h=-0.1, tau=4.0, eps=0.001, res=37, esphere=sphere5)
# u = net.u
# u_mem = net.u_mem
# zero = np.array([0.0,0.0,0.0])
# # obj_proj = []
# # for o in objects:
# #     obj_proj.append(sphere5.intersect(zero, o))
# while (t < T):
#     u,u_mem = net.step(dt, objects, objects[0], [],[])

    
# C5 = sphere5.C
# C6 = sphere6.C
# # k = 0
# # for i in range(sphere1.resol):
# #     for j in range(sphere1.resol):
# #         C5[i,j] = u[k]
# #         C6[i,j] = u_mem[k]
# #         k += 1
# sphere5.draw(C5)
# sphere6.draw(C6)
# # for o in objects:
# #     i,j = sphere6.getij(o)
# #     plot3DPoint(sphere6.ax, sphere6.getXYZ(i,j), 'red')
# # maxij = sphere6.resol - 1.0
# # ijcenter = sphere6.resol/2.0
# # obj_proj = []
# # # for o in objects:
# # #     obj_proj.append(sphere5.intersect(zero, o))
# # radToIndex = sphere6.resol/math.pi
# # for o in objects:
# #     i = min(round(math.atan2(o[1],o[2]) * radToIndex), maxij)
# #     j = min(round(math.atan2(o[0],o[1]) * radToIndex + ijcenter), maxij)
# #     print(i,j, sphere6.getXYZ(i,j))
# #     plot3DPoint(sphere6.ax, sphere6.getXYZ(i,j), 'red')

# fig.subplots_adjust(bottom=0.15)
# cbar_ax = fig.add_axes([0.25, 0.14, 0.5, 0.015])
# # plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, 
# #                     top=1.0, wspace=0.0,hspace=0.0)

# fig.colorbar(sphere1.scamap, cax=cbar_ax, location="bottom", orientation="horizontal")
# #fig.colorbar(self.scamap)


# # o = np.array([0.0,0.0,0.0])
# # d = np.array([-0.4,0.4,-0.3])
# # _, P2 = sphere4.intersect(o, d)
# # plot3DLine(ax[1,1], o, o+d, 'red')
# # plot3DPoint(ax[1,1], P2, 'black')
# plt.show()

# for j in range(sphere1.resol):
#     print(C[0,j])