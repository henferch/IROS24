import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
import numpy as np

class Gaussian:
    def __init__(self, sigma):
        sig = sigma
        SIGMA = np.array([[1.0,0.0],[0.0,1.0]])*sig
        self.inv_SIGMA = inv(SIGMA)
        self.denom = np.sqrt( ((2.0*np.pi)**2.0) * np.linalg.det(SIGMA) )
    def compute(self, K, MU):
        K_MU = K-MU
        return np.exp(-0.5* np.matmul(np.matmul(K_MU, self.inv_SIGMA), K_MU.transpose())) / self.denom            

class Sigmoid:
    def __init__(self, alpha):
        self.alpha = alpha
    def compute(self, k, mu):
        return 1.0/(1.0 + math.exp(self.alpha*(k-mu)))
    
class Sphere:
    def __init__(self, ax, res, title, cmap):
        self.ax = ax
        self.cmap = cmap
        self.title = title
        self.ax.set_title(self.title)
        self.resol = res
        self.maxij = self.resol - 1.0
        self.ijcenter = self.resol/2.0
        self.radToIndex = self.resol/math.pi
        theta, phi = np.linspace(0, np.pi, self.resol), np.linspace(0, np.pi, self.resol)
        THETA, PHI = np.meshgrid(theta, phi)
        self.s_radius = 0.4
        #self.s_radius_2 = self.s_radius**2.0  
        self.s_center = np.array([0.0,0.0,0.0])
        self.X = self.s_radius * np.sin(PHI) * np.cos(THETA)
        self.Y = self.s_radius * np.sin(PHI) * np.sin(THETA)
        self.Z = self.s_radius * np.cos(PHI)
        self.C = self.Y * 0.0
        self.scamap = plt.cm.ScalarMappable(cmap=self.cmap)
        self.refs = []
        for i in range(self.resol):
            for j in range(self.resol):
                self.refs.append(np.array([i,j]))
        self.refs = np.vstack(self.refs)

    def getRefs(self):
        return self.refs
    
    # def getXYZ(self, i, j):
    #     slpit_i = math.modf(i)
    #     slpit_j = math.modf(j)
    #     i_dec = slpit_i[0]
    #     j_dec = slpit_j[0]
    #     i_int = int(slpit_i[1])
    #     j_int = int(slpit_j[1])
    #     i_ = 1.0 - i_dec
    #     j_ = 1.0 - j_dec
    #     i_intp1 = i_int + 1
    #     j_intp1 = j_int + 1 
    #     x1 = self.X[i_int,j_int]
    #     x2 = self.X[i_intp1,j_int]
    #     x3 = self.X[i_int,j_intp1]
    #     x4 = self.X[i_intp1,j_intp1]
    #     x = ( (i_*x1 + (i_dec)*x2) + (j_*x3 + (j_dec)*x4) ) / 2.0
    #     y1 = self.Y[i_int,j_int]
    #     y2 = self.Y[i_intp1,j_int]
    #     y3 = self.Y[i_int,j_intp1]
    #     y4 = self.Y[i_intp1,j_intp1]
    #     y = ( (i_*y1 + (i_dec)*y2) + (j_*y3 + (j_dec)*y4) ) / 2.0
    #     z1 = self.Z[i_int,j_int]
    #     z2 = self.Z[i_intp1,j_int]
    #     z3 = self.Z[i_int,j_intp1]
    #     z4 = self.Z[i_intp1,j_intp1]
    #     z = ( (i_*z1 + (i_dec)*z2) + (j_*z3 + (j_dec)*z4) ) / 2.0
    #     return np.array([x,y,z])
    #     #return np.array([self.X[i,j],self.Y[i,j],self.Z[i,j]])
    
    def getij(self, p):
        p1 = p[1]
        i = math.atan2(p1,p[2]) * self.radToIndex - 1.0
        j = -math.atan2(p[0],p1) + np.pi/2.0
        j = j * self.radToIndex - 1.0
        
        return i,j

    def intersect(self, r_origin, r_dir):
        P1 = None
        P2 = None
        o = r_origin
        u = r_dir / np.linalg.norm(r_dir)
        c = self.s_center
        r = self.s_radius
        o_c = o-c
        dot_u_o_c = np.dot(u,o_c)
        delta = dot_u_o_c**2.0 - (np.linalg.norm(o_c)**2.0 - r**2.0)
        if delta > 0:
            delta_sqrt = delta**0.5
            d1 = -dot_u_o_c + delta_sqrt  
            P1 = o + u*d1
            d2 = -dot_u_o_c - delta_sqrt
            P2 = o + u*d2
        return P1,P2
     
    def draw(self, state):
        k = 0
        for i in range(self.resol):
            for j in range(self.resol):
                self.C[i,j] = state[k]
                k += 1

        self.fcolors = self.scamap.to_rgba(self.C)
        self.ax.plot_surface(self.X, self.Y, self.Z, facecolors=self.fcolors, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.7)
        #fig.colorbar(self.scamap)
        self.ax.view_init(elev=0, azim=89, roll=0)
        self.ax.set_aspect('equal')
        self.ax.grid(False)
        self.ax.axis('off')

    def plot3DLine(self, ax, P1, P2, color):
        return ax.plot3D([P1[0],P2[0]], [P1[1],P2[1]], [P1[2],P2[2]], color=color, linestyle="dashed", linewidth=0.5)[0]
    
    def plot3DPoint(self, ax, P1, color):
        return ax.plot3D(P1[0], P1[1], P1[2], color=color, marker='o', markersize=10.0)[0]
