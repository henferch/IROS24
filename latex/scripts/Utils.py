#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import numpy as np
import math
import random
import copy
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class Utils:
    ZEROVECTOR = np.array([1.0,0.0,0.0,1.0], dtype=np.float32)
    __instance = None
    
    @staticmethod 
    def getInstance():
       """ Static access method. """
       if Utils.__instance == None:
          Utils()
       return Utils.__instance
   
    def boltzmann(self, op1, op2, sign=1.0):
        x = []
        alpha = 100.0*sign
        for o1, o2 in zip(op1, op2):
            exp_ao1 = math.exp(o1*alpha)
            exp_ao2 = math.exp(o2*alpha)
            x.append( (o1*exp_ao1 + o2*exp_ao2)/(exp_ao1 + exp_ao2) ) 
        return x

    def maxBoltzmann(self, op1, op2):
        return self.boltzmann(op1, op2)
    
    def minBoltzmann(self, op1, op2):
        return self.boltzmann(op1, op2,-1.0)

    def getCurrentTimeMS(self):    
        return int(round(time.time() * 1000))        

    def getParameters(self, fname_):
        params = None
        with open(fname_) as json_file:
            params = json.load(json_file)
        return params
        
    def __init__(self):
        """ Virtually private constructor. """
        if Utils.__instance != None:
           raise Exception("This class is a singleton class, use 'getInstance()' instead!")
        else:                    
            random.seed(7)
            np.random.seed(7)
            self._t0 = self.getCurrentTimeMS()
            self._uniqueNetIds = {}
            self._eps = 1.0e-30
            Utils.__instance = self
                        
    def getEpsilon(self):        
        return self._eps
    
    def getXYRotHom(self, yaw_, pitch_):
        cy = np.cos(yaw_) 
        sy = np.sin(yaw_)
        cp = np.cos(pitch_) 
        sp = np.sin(pitch_)
        R = np.array([[cy*cp, -sy, cy*sp, 0.0],[sy*cp, cy, sy*sp, 0.0], [-sp, 0.0, cp, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        return R
        
    def get3DOrientationfromYawPitch(self, y_, p_):
            R = self.getXYRotHom( y_, p_)       
            v = np.dot(R, self.__class__.ZEROVECTOR)[0:3]
            return v
        
        
    def computeStateBelief(self, h_, b_, g_):    
        y = h_
        g = g_
        if not (b_ is None or g is None):
            if g > 1.0:
                g = 1.0
            elif g < 0.0:
                g = 0.0
            y = (1.0 - g)*h_ + g*b_                
        return y
    
    def saveDict(self, dict_, fName_):
        f = open("{}.json".format(fName_),"w")  
        jsonData = json.dumps(dict_, cls=NumpyEncoder)            
        f.write(jsonData)                       
        f.close()                               

    def loadDict(self, fName_):
        f = open("{}.json".format(fName_),"rb")   
        jsonData = json.load(f)                     
        f.close()
        return jsonData                               
        
    def unregisterNetworkId(self, netId_):
        if netId_ in self._uniqueNetIds:
            self._uniqueNetIds.pop(netId_)

    def unregisterLayerId(self, netId_, id_):
        if netId_ in self._uniqueNetIds:
            if id_ in self._uniqueNetIds[netId_]:
                self._uniqueNetIds[netId_].remove(id_)

    def getMultiGaussianConst(self, sigma_):
        k = sigma_.shape[0]    
        sigma = sigma_.reshape((k,-1))
        invSigma = np.linalg.inv(sigma)
        den = np.sqrt((2.0*np.pi)**k + np.abs(np.linalg.det(sigma)))                
        if sigma_.shape[1] == 1:
            invSigma = invSigma.squeeze()
        return invSigma, den
        
    def normalize(self, v_):
        norm = np.linalg.norm(v_) + self._eps
        return v_/norm
    
    def dotProduct(self, x_, y_):
        return np.dot(self.normalize(x_),self.normalize(y_))
                      
    def truncateScalar(self, s_, li_=0.0, ls_=1.0):
        if s_ < li_:
            return li_
        if s_ > ls_:
            return ls_
        return s_
    
    def multiGaussianPrecomp(self, mu_, v_, invSigma_, den_):     
        g = []
        v = np.array(v_)
        n = mu_.shape[0]
        mu = mu_.reshape((n,-1))
        for i in range(n):
            diff = np.array(mu[i,:]) - v
            g.append((np.exp(-0.5 * np.dot(np.dot(diff, invSigma_), diff)) / den_))
        return np.array(g)
            
    def multiGaussian(self, mu_, v_, sigma_):
        invSigma = np.linalg.inv(sigma_)
        sigmaDet = np.linalg.det(sigma_)
        k = sigma_.shape[0]*1.0
        den = np.sqrt( (2.0*np.pi)**k + np.abs(sigmaDet))    
        return self.multiGaussianPrecomp(mu_, v_, invSigma, den)        

    def softmax(self, x_):
        ex = np.exp(x_)
        return ex/ex.sum()

    """ 
    x: data
    c: centroid where f(x) = 0.5
    l: transition slope 
    """
    def sigmoid(self, x_, c_=0.0, l_=1.0):
        return 1.0/(1.0+np.exp(-l_*(x_-c_)))         
    
    def getLinRef(self, min_, max_, n_):
        mu = [min_]
        inter = max_ - min_
        step = inter/(n_ - 1.0)
        for i in range(n_-2):
            mu.append(mu[-1]+step)
        mu.append(max_)
        return np.array(mu)
        
    def getMeanRef(self, ref_, state_):
        return np.dot(ref_, state_/state_.sum())
    
    def getMaxRef(self, ref_, state_):
        return ref_[np.argmax(state_)]
    
    def setPlotData3DLine(self, obj, P1, P2):
        obj.set_data([P1[0],P2[0]], [P1[1],P2[1]])
        obj.set_3d_properties([P1[2],P2[2]])

    def setPlotData3DPoint(self, obj, P1):
        obj.set_data(P1[0], P1[1])
        obj.set_3d_properties(P1[2])
    
    def plot3DLine(self, ax, P1, P2, color, linewidth):
        return ax.plot3D([P1[0],P2[0]], [P1[1],P2[1]], [P1[2],P2[2]], color=color, linewidth=linewidth)[0]

    def plot3DPoint(self, ax, P1, color):    
        return ax.plot([P1[0]], [P1[1]], [P1[2]], color=color, marker='o')[0]    




