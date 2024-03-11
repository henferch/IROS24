import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from AEGO.Utils import Utils
import copy

class GeodesicDome:
    """
    This class modified and adjusts the implementation of a geodome provided by 
    https://github.com/nbhr/geodome
    (the projest is subject to MIT License: Copyright (c) 2020 nbhr)  

    Geodesic Dome of Nv vertices and Nf faces.

    self.v: (Nv, 3) array. list of vertices.
    self.f: (Nf, 3) array. list of vertex indices to define faces.
    """
    def __init__(self, params):
        self._ut = Utils.getInstance()    

        self.tol = 1e-15 

        self. v, self.f = self.buildStructure()
        
        self.center = params['center']

        self.tesselation = params['tesselation']
        
        # Subdivide N times
        self.tessellate(self.tesselation)

        self.scale_factor = params['radius']
        self.scale(self.scale_factor)
        
        self.v_to_f = []

        for v in range(len(self.v)) :
            faces = []
            for f in range(len(self.f)):
                if v in self.f[f]:
                    faces.append(f)
            self.v_to_f.append(faces) 
            
        self.v_N = len(self.v)
        self.f_N = len(self.f)

        self._renderObj = None
        self.scamap = None
        self.cmap = "viridis"

        # # Check the number of vertices / faces
        print('num of vertices = {}, num of faces = {}'.format(self.v_N, self.f_N))

        
    def buildStructure(self):
        ## vertices ##
        p = (1.0 + np.sqrt(5.0)) / 2.0
        a = np.sqrt((3.0 + 4.0 * p) / 5.0) / 2.0
        b = np.sqrt(p / np.sqrt(5.0))
        c = np.sqrt(3.0 / 4.0 - a ** 2.0)
        d = np.sqrt(3.0 / 4.0 - (b - a)** 2.0)
        # icosahedron in (r, theta, z) == cylindrical coordinates
        v = np.array([
            [0.0, 0.0, (c + d / 2.0)],
            [b, 2.0*0.0*np.pi / 5.0, d / 2.0],
            [b, 2.0*1.0*np.pi / 5.0, d / 2.0],
            [b, 2.0*2.0*np.pi / 5.0, d / 2.0],
            [b, 2.0*3.0*np.pi / 5.0, d / 2.0],
            [b, 2.0*4.0*np.pi / 5.0, d / 2.0],
            [b, (2.0*0.0+1.0)*np.pi / 5.0, - d / 2.0],
            [b, (2.0*1.0+1.0)*np.pi / 5.0, - d / 2.0],
            [b, (2.0*2.0+1.0)*np.pi / 5.0, - d / 2.0],
            [b, (2.0*3.0+1.0)*np.pi / 5.0, - d / 2.0],
            [b, (2.0*4.0+1.0)*np.pi / 5.0, - d / 2.0],
            [0.0, 0.0, -(c + d / 2.0)],
        ])
        # icosahedron in (x, y, z) == Cartesian coordinates
        v = np.vstack([
            v[:, 0]*np.cos(v[:, 1]),
            v[:, 0]*np.sin(v[:, 1]),
            v[:, 2]
        ]).T
        # normalize the radius
        v *= (1.0 / v[0, 2])

        # fix super small values to zero        
        v[np.abs(v) < self.tol] = 0        

        ## faces ##
        f = np.array([
            [2, 0, 1],
            [3, 0, 2],
            [4, 0, 3],
            [5, 0, 4],
            [1, 0, 5],
            [2, 1, 6],
            [7, 2, 6],
            [3, 2, 7],
            [8, 3, 7],
            [4, 3, 8],
            [9, 4, 8],
            [5, 4, 9],
            [10, 5, 9],
            [6, 1, 10],
            [1, 5, 10],
            [6, 11, 7],
            [7, 11, 8],
            [8, 11, 9],
            [9, 11, 10],
            [10, 11, 6],
        ])
        return v, f

    def getV(self):
        return copy.copy(self.v)
    
    def scale(self, factor):
        self.v = self.v * factor

    def intersect(self, origin, dir):
        P1 = None
        P2 = None
        o = origin
        u = dir / np.linalg.norm(dir)
        c = self.center
        r = self.scale_factor
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

    def plot(self, ax, state, alpha=0.4, showVertex=False, showEdges=False):                
        face_act = np.zeros((self.f_N), dtype=np.float32)
        for v in range(self.v_N):
            face_act[self.v_to_f[v]] += state[v]
        centered = self.v + self.center
        self.tri = Poly3DCollection(centered[self.f])
        if showEdges:
            self.tri.set_edgecolor('gray')
            self.tri.set_linewidth(0.5)
        self.tri.set_alpha(alpha)
        scamap = plt.cm.ScalarMappable(cmap=self.cmap)
        fcolors = scamap.to_rgba(face_act)
        self.tri.set_facecolors(fcolors)        
        self._renderObj = ax.add_collection3d(self.tri)
        if showVertex:
            for v in self.v:
                self._ut.plot3DPoint(ax, v,'gray', 'o', 1.0)
            

    def tessellate(self, iter=1):
        def newvert(v0, v1):
            v = v0 + v1
            v /= np.linalg.norm(v)
            return v

        for _ in range(iter):
            f = self.f
            v = self.v
            v2 = []
            vv2v = {}
            vid = len(v)
            for tri in self.f:
                for i, j in zip([0, 1, 2], [1, 2, 0]):
                    if tri[i] < tri[j]:
                        vv2v[tri[i], tri[j]] = vv2v[tri[j], tri[i]] = vid
                        vid += 1
                        v2.append(newvert(v[tri[i]], v[tri[j]]))
            v = np.vstack([v, np.array(v2)])

            f2 = []
            for tri in self.f:
                f2.append([tri[0], vv2v[tri[0], tri[1]], vv2v[tri[2], tri[0]]])
                f2.append([tri[1], vv2v[tri[1], tri[2]], vv2v[tri[0], tri[1]]])
                f2.append([tri[2], vv2v[tri[2], tri[0]], vv2v[tri[1], tri[2]]])
                f2.append([vv2v[tri[0], tri[1]], vv2v[tri[1], tri[2]], vv2v[tri[2], tri[0]]])

            self.v = v
            self.f = np.array(f2)

        self.v[np.abs(self.v) < self.tol] = 0        
        return self
        
    def face_normal(self):
        """
        This function is not needed in most cases, since the vertex position is identical to its normal.
        """
        tri = self.v[self.f]
        n = np.cross(tri[:, 1,:] - tri[:, 0,:], tri[:, 2,:] - tri[:, 0,:])
        n /= np.linalg.norm(n, axis=0)
        return n

    def save_as_ply(self, filename):
        with open(filename, 'w') as fp:
            fp.write('ply\n')
            fp.write('format ascii 1.0\n')
            fp.write('element vertex {}\n'.format(len(self.v)))
            fp.write('property float x\n')
            fp.write('property float y\n')
            fp.write('property float z\n')
            fp.write('element face {}\n'.format(len(self.f)))
            fp.write('property list uchar int vertex_indices\n')
            fp.write('end_header\n')
            np.savetxt(fp, self.v)
            f2 = np.hstack([(np.ones(len(self.f), dtype=np.int32) * 3).reshape((-1, 1)), self.f])
            np.savetxt(fp, f2, fmt='%d')

    def render(self, state):
        #self._renderObj.remove()
        self.scamap = plt.cm.ScalarMappable(cmap='viridis')
        face_act = np.zeros((self.f_N), dtype=np.float32)
        for v in range(self.v_N):
            face_act[self.v_to_f[v]] += state[v]
        fcolors = self.scamap.to_rgba(face_act)        
        self.tri.set_facecolors(fcolors)
        #self._renderObj = ax.add_collection3d(self.tri)                    
            
# # Icosahedron
# tesselation = 3    
# scale = 0.4
# sigma = 0.001
# objects = [np.array([0.4,0.0,0.0]), np.array([0.0,0.4,0.0])]

# params = {'tesselation': tesselation, 'scale' : scale, 'objects': objects, 'sigma': sigma}
        
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10, 10))        
# fig.tight_layout()
# ax.view_init(elev=0, azim=0)

# g = GeodesicDome(params)

# g.updateState()

# objEgoSph = {'obj' : g.plot(ax, g.face_act)}

# ax.set_aspect('equal')
# ax.grid(False)
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)

# timer = fig.canvas.new_timer(interval=100)
# timer.add_callback(g.render, ax, objEgoSph, g.face_act)
# timer.start()

# plt.show()