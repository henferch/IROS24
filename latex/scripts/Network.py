import numpy as np
import copy

class Network:
    def __init__(self, p_= None) :
        self._ut = p_['ut']    
        self._ref = p_['ref']                   # attractor reference values
        self._N = self._ref.shape[0]            # attractor unit number
        self._res = p_['res']                   # angle resolution 
        self._sigma = np.array(p_['sig'])              
        self._h_pre = p_['h_pre']               # rest point h
        self._h_sel = p_['h_sel']               # rest point h
        self._inh = p_['inh']                   # inhibition factor
        self._dt = p_['dt']                     # time step in ms
        self._tau = (p_['tau']) 
        self._objects = p_['objects']
        self._u_sel_u_pre_gain = 2.5
        self._u_sel_sig_l = 100.0
        self._nObjects = len(self._objects)

        # precomputed const 
        self._invSigma, self._den = self._ut.getMultiGaussianConst(self._sigma)
        
        Wo = [] 
        for o in self._objects:
            Wo.append(self._ut.multiGaussianPrecomp(self._ref, o, self._invSigma, self._den))
        if len(Wo) > 0:
            self._W_obj = np.vstack(Wo)
        else:
            self._W_obj = None

        self._W_pre = np.zeros((self._N,self._N),dtype=np.float32) # recurrent weights
        #self._W_pre_right = np.zeros((self._N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        
        #self._W_pre_left = np.zeros((self._N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        
        #self._W_pre_above = np.zeros((self._N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        
        #self._W_pre_below = np.zeros((self._N,),dtype=np.float32) # symetry in vertical axis, so only ine row is needed        

        self._W_sel = np.zeros((self._N,self._N),dtype=np.float32) # recurrent weights        
                        
        # weights for u_pre
        for i in range(self._N):
            self._W_pre[i,:] = self._ut.multiGaussianPrecomp(self._ref, self._ref[i,:], self._invSigma, self._den)            

        # # weights for u_pre_left
        # mu =  self._res/2.0 + self._res/20.0
        # k = 0
        # for i in range(self._res):
        #     for j in range(self._res):
        #         self._W_pre_left[k] = self._ut.sigmoid(np.array([mu - j]),0, 2.0)[0]
        #         k += 1
        
        # # weights for u_pre_right
        # mu =  self._res/2.0 - self._res/20.0
        # k = 0
        # for i in range(self._res):
        #     for j in range(self._res):
        #         self._W_pre_right[k] = self._ut.sigmoid(np.array([j - mu]),0, 2.0)[0]
        #         k += 1

        # weights for u_sel
        for i in range(self._N):            
            mgpc = self._ut.multiGaussianPrecomp(self._ref, self._ref[i,:], self._invSigma, self._den)
            mgpc /= mgpc.max()
            self._W_sel[i,:] = mgpc - 1.0
        
        #print(self._W_sel)
        self._W_pre_inh = self._W_pre + self._inh
        
        self._u_pre = np.random.normal(0,0.01,self._N)*self._h_pre
        self._u_sel = -np.random.normal(0.8,0.01,self._N)   
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
        
        # right 
        right = input_['r']
        if right > 0.0:
            MU = 1.0
            for i in range(self._nObjects):
                MU += self._o[i]*self._objects[i][1]            
            k = 0
            for i in range(self._res):
                for j in range(self._res):
                    self._W_pre_right[k] = self._ut.sigmoid(np.array([j - MU]),0, 5.0)[0]
                    k += 1
            du_pre += right*self._W_pre_right * self._u_pre        

        # left 
        left = input_['l']
        if left > 0.0:
            MU = -1.0
            for i in range(self._nObjects):
                MU += self._o[i]*self._objects[i][1]            
            k = 0
            for i in range(self._res):
                for j in range(self._res):
                    self._W_pre_left[k] = self._ut.sigmoid(np.array([MU - j]),0, 5.0)[0]
                    k += 1
            du_pre += left*self._W_pre_left * self._u_pre

        # above 
        above = input_['a']
        if above > 0.0:
            MU = -1.0
            for i in range(self._nObjects):
                MU += self._o[i]*self._objects[i][0]            
            k = 0
            for i in range(self._res):                
                for j in range(self._res):                    
                    self._W_pre_above[k] = self._ut.sigmoid(np.array([MU-i]),0, 5.0)[0]
                    k += 1
            du_pre += above*self._W_pre_above * self._u_pre

        # below
        below = input_['b']
        if below > 0.0:
            MU = 1.0
            for i in range(self._nObjects):
                MU += self._o[i]*self._objects[i][0]            
            k = 0
            for i in range(self._res):                
                for j in range(self._res):                    
                    self._W_pre_below[k] = self._ut.sigmoid(np.array([i-MU]),0, 5.0)[0]
                    k += 1
            du_pre += below*self._W_pre_below * self._u_pre

        # next
        next = input_['n']
        if next > 0.0:
            du_pre += -next*(self._ut.softmax(50.0*self._u_sel))

        # pre-selection
        self._u_pre += self._dt*du_pre/self._tau


        # selection
        noise = np.random.normal(0,0.01,self._N)
        s_sel = self._u_sel + ( self._u_sel_u_pre_gain * self._u_pre )
        s_sel_act = self._ut.sigmoid(s_sel, 0.0, self._u_sel_sig_l)
        du_sel = -self._u_sel + np.matmul(self._W_sel, s_sel_act) + self._h_sel + noise
        self._u_sel += self._dt*du_sel/self._tau

        # object layer
        if not self._W_obj is None:
            self._o = self._ut.softmax(50.0 * np.matmul(self._W_obj,  self._u_sel))
        
        return copy.copy(self._u_pre), copy.copy(self._u_sel), copy.copy(self._o)
    
    def getU_pre(self):
        return copy.copy(self._u_pre)
    
    def getU_sel(self):
        copy.copy(self._u_sel), copy.copy(self._o)