import numpy as np
import copy

class NetworkGeodesic:
    def __init__(self, p_= None) :
        self._ut = p_['ut']    
        self._ref = p_['ref']                   # attractor reference values
        self._N = self._ref.shape[0]            # attractor unit number        
        self._sigma = np.array(p_['sig'])              
        self._h_pre = p_['h_pre']               # rest point h
        self._h_sel = p_['h_sel']               # rest point h
        self._inh = p_['inh']                   # inhibition factor
        self._dt = p_['dt']                     # time step in ms
        self._tau_pre = p_['tau_pre']
        self._tau_sel = p_['tau_sel'] 
        self._objects = p_['objects']
        self._o_alpha = p_['o_alpha']
        self._u_sel_u_pre_gain = 2.5
        #self._u_sel_u_pre_gain = 2.0
        #self._u_sel_u_pre_gain = 2.0
        #self._u_sel_sig_l = 100.0
        self._u_sel_sig_l = 100.0
        self._nObjects = len(self._objects)

        # precomputed const 
        self._invSigma, self._den = self._ut.getMultiGaussianConst(self._sigma)

        self._sigmaLR = np.eye(3)*0.5
        self._sigmaLR[2,2] = 0.008
        self._invSigmaLR, self._denLR = self._ut.getMultiGaussianConst(self._sigmaLR)

        self._sigmaAB = np.eye(3)*0.5
        self._sigmaAB[1,1] = 0.008
        self._invSigmaAB, self._denAB = self._ut.getMultiGaussianConst(self._sigmaAB)

        Wo = [] 
        for o in self._objects:
            Wo.append(self._ut.multiGaussianPrecomp(self._ref, o, self._invSigma, self._den))
        if len(Wo) > 0:
            self._W_obj = np.vstack(Wo)
        else:
            self._W_obj = None

        self._W_pre = np.zeros((self._N,self._N),dtype=np.float32) # recurrent weights
        self._W_pre_right = np.zeros((self._N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        
        self._W_pre_left = np.zeros((self._N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        
        self._W_pre_above = np.zeros((self._N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        
        self._W_pre_below = np.zeros((self._N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        

        self._offset = 0.00
        self._W_sel = np.zeros((self._N,self._N),dtype=np.float32) # recurrent weights        
                        
        # weights for u_pre
        for i in range(self._N):
            self._W_pre[i,:] = self._ut.multiGaussianPrecomp(self._ref, self._ref[i,:], self._invSigma, self._den)            

        # weights for u_sel
        for i in range(self._N):            
            mgpc = self._ut.multiGaussianPrecomp(self._ref, self._ref[i,:], self._invSigma, self._den)
            mgpc /= mgpc.max()
            self._W_sel[i,:] = mgpc - 1.0
        
        #print(self._W_sel)
        self._W_pre_inh = self._W_pre + self._inh
        
        self._u_pre = np.random.normal(0,0.01,self._N)*self._h_pre
        self._u_sel = -np.random.normal(0.1,0.01,self._N)   
        if self._nObjects > 0:              
            self._o = np.zeros((self._nObjects,), dtype=np.float32)
        else:
            self._o = None
            
    def updateObjects(self, objects):
        if len(objects) != self._nObjects:
            raise Exception('the same number of tracked objects must be updated')
        self._objects = objects
        Wo = [] 
        for o in self._objects:
            Wo.append(self._ut.multiGaussianPrecomp(self._ref, o, self._invSigma, self._den))
        self._W_obj = np.vstack(Wo)

    def step(self, input_):
        # du_pre
        du_pre = -self._u_pre + np.matmul(self._W_pre_inh, self._u_pre) + self._h_pre
        
        # input 
        inputObj = input_['o']
        for i in range(self._nObjects):
            du_pre += inputObj[i]*self._W_obj[i,:] 
        
        right = input_['r']
        if right > 0.0:
            MU = 0.0
            for i in range(self._nObjects):
                MU += self._o[i]*self._objects[i]
            self._W_pre_right = self._ut.sigmoid(self._ref[:,1] - MU[1], 0.0, 50.0) * self._ut.multiGaussianPrecomp(MU, self._ref, self._invSigmaLR, self._denLR)                
            du_pre += right*self._W_pre_right * self._u_pre

        left = input_['l']
        if left > 0.0:
            MU = 0.0
            for i in range(self._nObjects):
                MU += self._o[i]*self._objects[i]
            self._W_pre_left = self._ut.sigmoid(MU[1] - self._ref[:,1],0.0, 50.0) * self._ut.multiGaussianPrecomp(MU, self._ref, self._invSigmaLR, self._denLR)                
            du_pre += left*self._W_pre_left * self._u_pre
        
        above = input_['a']
        if above > 0.0:
            MU = 0.0
            for i in range(self._nObjects):
                MU += self._o[i]*self._objects[i]
            self._W_pre_above = self._ut.sigmoid(self._ref[:,2] - MU[2], 0.0, 50.0) * self._ut.multiGaussianPrecomp(MU, self._ref, self._invSigmaAB, self._denAB)                
            du_pre += above*self._W_pre_above * self._u_pre

        below = input_['b']
        if below > 0.0:
            MU = 0.0
            for i in range(self._nObjects):
                MU += self._o[i]*self._objects[i]
            self._W_pre_below = self._ut.sigmoid(MU[2] - self._ref[:,2], 0.0, 50.0) * self._ut.multiGaussianPrecomp(MU, self._ref, self._invSigmaAB, self._denAB)
            du_pre += below*self._W_pre_below * self._u_pre

        # next
        next = input_['n']
        if next > 0.0:
            #du_pre += -next*(self._ut.softmax(15.5*self._u_sel))
            du_pre += -next*(self._ut.softmax(15.5*self._u_sel))

        # pre-selection
        self._u_pre += self._dt*du_pre/self._tau_pre

        # selection
        s_noise = np.random.normal(0,0.001,self._N)
        #s_noise = np.random.normal(0,0.001,self._N)
        s_sel = self._u_sel + ( self._u_sel_u_pre_gain * self._u_pre )
        s_sel_act = self._ut.sigmoid(s_sel, 0.0, self._u_sel_sig_l)
        du_sel = -self._u_sel + np.matmul(self._W_sel, s_sel_act) + self._h_sel + s_noise
        self._u_sel += self._dt*du_sel/self._tau_sel
        
        # s_stim = self._u_sel_u_pre_gain * self._u_pre
        # s_sel_act = self._ut.sigmoid(self._u_sel, 0.0, self._u_sel_sig_l)
        # du_sel = -self._u_sel + np.matmul(self._W_sel, s_sel_act) + s_stim + self._h_sel + s_noise
        # self._u_sel += self._dt*du_sel/self._tau_sel

        # object layer
        if not self._W_obj is None:
            self._o = self._ut.softmax(self._o_alpha * np.matmul(self._W_obj,  self._u_sel))
        
        return copy.copy(self._u_pre), copy.copy(self._u_sel), copy.copy(self._o)
    
    def getU_pre(self):
        return copy.copy(self._u_pre)
    
    def getU_sel(self):
        copy.copy(self._u_sel), copy.copy(self._o)