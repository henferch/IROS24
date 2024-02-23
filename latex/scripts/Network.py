import numpy as np
from Utils import Utils
import copy

class Network:
    def __init__(self, p_= None) :
        self._ut = Utils.getInstance()    
        self._ref = p_['ref']                   # attractor reference values
        self._N = self._ref.shape[0]            # attractor unit number
        self._sigma = np.array(p_['sig'])              
        self._h = p_['h']                       # rest point h
        self._inh = p_['inh']                   # inhibition factor
        self._dt = p_['dt']                     # time step in ms
        self._tau = (p_['tau']) 
        #self._one_Min_TAU = 1.0 - self._TAU
        
        self._W = np.zeros((self._N,self._N),dtype=np.float32) # recurrent weights
        self._W_mem = np.zeros((self._N,self._N),dtype=np.float32) # recurrent weights        
                
        # precomputed const 
        self._invSigma, self._den = self._ut.getMultiGaussianConst(self._sigma)
                
        # setting recurrent weights analytically
        for i in range(self._N):
            self._W[i,:] = self._ut.multiGaussianPrecomp(self._ref, self._ref[i,:], self._invSigma, self._den)            
            #mgpc = self._ut.multiGaussianPrecomp(self._ref, self._ref[i,:], self._invSigma, self._den)
            #self._W[i,:] = mgpc/mgpc.sum()

        # setting recurrent memory weights
        for i in range(self._N):            
            mgpc = self._ut.multiGaussianPrecomp(self._ref, self._ref[i,:], self._invSigma, self._den)
            mgpc /= mgpc.max()
            self._W_mem[i,:] = mgpc - 1.0
            # # winh = np.ones((self._N,),dtype=np.float32)/(-self._N*1.0)
            # winh = np.ones((self._N,),dtype=np.float32)*(-1.0)
            # winh[i] = 0.0
            # self._W_mem[i,:] = winh        
        
        #print(self._W_mem)
        self._W_inh = self._W + self._inh
        
        self._u = np.random.normal(0,0.01,self._N)*self._h
        self._u_mem = -np.random.normal(0.1,0.01,self._N)
                    
    def step(self, input_):
        du = -self._u + np.matmul(self._W_inh, self._u) + self._h
        imp = np.zeros((self._N,),dtype=np.float32)
        for i in input_:
            imp += self._ut.multiGaussianPrecomp(self._ref, i, self._invSigma, self._den)
        du += imp
        self._u += self._dt*du/self._tau

        noise = np.random.normal(0,0.01,self._N)
        du_mem = -self._u_mem + np.matmul(self._W_mem, self._ut.sigmoid(self._u_mem, 0.0, 100.0)) + self._h + noise
        # print('du_mem', du_mem)
        # print('imp', imp)
        du_mem += self._u*0.5
        self._u_mem += self._dt*du_mem/self._tau

        return copy.copy(self._u), copy.copy(self._u_mem)