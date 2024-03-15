import numpy as np
import cv2
import posix_ipc as pos
import json
import struct
import mmap
import sys
import math
from AEGO.NetworkGeodesic import NetworkGeodesic
from AEGO.GeodesicDome import GeodesicDome
from AEGO.Robot import RobotViewer
from AEGO.Human import HumanViewer
from AEGO.Utils import Utils
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image
import os

print("program start")

def loadFrames(exp):
    url = '/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/'.format(exp)
    files = os.listdir(url)    
    frames = []
    for f in files :
        if 'png' in f:
            frames.append(f)
    num = []
    fDict = {}
    for f in frames :
        t = str(f)
        i = int(f.split('_')[1].split('.')[0])
        num.append(i)
        fDict[i] = t
    num.sort()
    finenames = []
    for i in num:
        finenames.append(fDict[i])

    return finenames

# Opening JSON file
params = None

try:
    f = open('/home/hfchame/Workspace/VSCode/IROS24/src/parameters.json')
    params = json.load(f)
    f.close()
except Exception as ex:
    print ("Can't load parameters from file 'parameters.json'. Error: {}".format(ex))
    sys.exit(1)

print("program start")
# Opening JSON file
params = None
try:
    f = open('/home/hfchame/Workspace/VSCode/IROS24/src/parameters.json')
    params = json.load(f)
    f.close()
except Exception as ex:
    print ("Can't load parameters from file 'parameters.json'. Error: {}".format(ex))
    sys.exit(1)
    
user = params["expID"] 
dt = params["dt"]

ExpID = range(5,12)
ut = Utils.getInstance()

objects = {'64L': np.array([1.19082725, 0.45242429, 1.05075765]),\
            '80L': np.array([ 1.91267562, -0.73908919,  1.09514868]), 
            '68L': np.array([ 1.27697361, -0.09755165,  1.06898379]), 
            '84L': np.array([1.67314959, 0.58602697, 1.53109407])}

oKeys  = ['68L', '84L', '64L', '80L']

I = [340, 387, 297, 247, 269, 257, 283]
F = [1676, 1455, 1317, 1267, 1109, 1104, 1344]

fps = 10
timeSleep = int(1000.0/fps)

pNn = params['NeuralNetwork']
stop = False
#for e in range(5):

all_h_human = []
for e in [5]:

    
    print("processisng experiment ".format(e))
    exp = ExpID[e]
    
    dataPath = '/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/'.format(exp)
    keyFrameSeg = loadFrames(exp)        

    humanPose = None
    with open('{}humanPose.npy'.format(dataPath), 'rb') as f:
        humanPose = np.load(f)

    robotPose = None
    with open('{}robotPose.npy'.format(dataPath), 'rb') as f:
        robotPose = np.load(f)
        robotPose = robotPose.reshape((humanPose.shape[0],14,3))
    
    dataRobot = None
    with open('{}robotDemo.npy'.format(dataPath), 'rb') as f:
        dataRobot = np.load(f)

    dataHuman = None
    with open('{}humanDemo.npy'.format(dataPath), 'rb') as f:
        dataHuman = np.load(f)        
        dataHuman = dataHuman.reshape((dataRobot.shape[0], 16, 3))    

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(8, 4))
    fig.tight_layout()

    robot = RobotViewer(ut, axs[0])
    human = HumanViewer(ut, axs[1])

    #robot.setFrames(dataRobot[0,:,:])    
    robot.setFrames(robotPose[0,:])
    robotBaseFramePos = robot.getBaseLocation()

    #dataHuman[:,0] *= -1.0
    dataHuman[:,:,1] *= -1.0
    human.setBaseFrame(np.array([0.0, 0.0, params['humanBaseFramePos'][2]]))
    human.setFrames(dataHuman[0,:,:])
    humanBaseFramePos = human.getBaseLocation()

    robot.plot('darkcyan', 1.0)
    human.plot('darkcyan', 1.0)

    egoSphereRobot = None
    networkRobot = None
    egoSphereHuman = None
    networkHuman = None
    objs_int = []
    for n, pNn, bf, ax in zip( ['r', 'h'], 
                            [pNn['robot'], pNn['human']],\
                            [robotBaseFramePos, humanBaseFramePos],
                            axs):
        pNnGd = pNn['GeoDome']
        pNnGd['center'] = bf 
        ego = GeodesicDome(pNnGd)
        if n == 'h':
            # for the human 2 points representing both forearms intersections 
            pNn['objects'] = [bf, bf]
        else:
            # for the robot 4 points representing objects
            #objs_int = []
            for oK in oKeys:
                _, p = ego.intersect(ego.center, ego.center-objects[oK])                                                
                objs_int.append(p-ego.center)
                #ut.plot3DPoint(ax, p, 'red')
            pNn['objects'] = objs_int
        pNn['ut'] = ut
        pNn['sigma'] = pNn['sigma']*np.eye(3, dtype=np.float32)
        pNn['dt'] = params['dt']
        pNn['ref'] = ego.getV()
        net = NetworkGeodesic(pNn)
        if n == 'r':
            egoSphereRobot = ego
            networkRobot = net
        else:
            egoSphereHuman = ego
            networkHuman = net    
        ego.plot(ax, net.getU_pre())        

    #humanHandWeight = np.zeros((2,), dtype=np.float32)

    # Hebbian learning    
    W_hr = np.zeros((egoSphereRobot.v_N,egoSphereHuman.v_N), dtype=np.float32)

    for o in range(len(oKeys)) :
        ob = objects[oKeys[o]]
        ib = objs_int[o] + robot.getBaseLocation()
        ut.plot3DLine(axs[0], ob, ib, 'orangered', 1.0, 'dashed')
        ut.plot3DPoint(axs[0], ob, 'black')
    
    for o in range(len(oKeys)):
    #for o in [0]:
        
        u_hebbRobot = networkHuman.getU_pre() * 0.0

        #robot.setFrames(dataRobot[o,:])        
        human.setFrames(dataHuman[o,:])

        # urlImg = '{}{}'.format(dataPath, keyFrameSeg[o])
        # print(urlImg)
        # img = np.asarray(Image.open(urlImg))
        # axs[2].imshow(img) 

        Torso, RElbow, RWrist, LElbow, LWrist = human.getIntersectionPoints()
        pR1, pR2 = ego.intersect(RElbow, RWrist-RElbow)
        pL1, pL2 = ego.intersect(LElbow, LWrist-LElbow)      
              
        if not (pR1 is None or pL1 is None):  
            # ut.plot3DPoint(axs[1], pR1, 'red')
            # ut.plot3DPoint(axs[1], pL1, 'red')
            networkHuman.updateObjects([pR1-Torso, pL1-Torso])
            iHArm = 0                
            if pR1[2] < pL1[2]:
                iHArm = 1

        h = networkHuman._W_obj[iHArm,:]
        all_h_human.append(h)
        for i in range(W_hr.shape[0]):
            W_hr[i,:] += networkRobot._W_obj[o,i]*h
        
        u_hebbRobot += np.matmul(W_hr, h)
        
        # plot hebbian synaptic weights
        # fig = plt.figure(2)
        # plt.tight_layout()
        # im1 = plt.imshow(W_hr, interpolation='nearest', aspect='auto')

        human.render()
        robot.render()
        egoSphereRobot.render(u_hebbRobot)
        egoSphereHuman.render(h)

        
    #print(networkRobot._W_obj[o,:])
    # egoSphereRobot.plot(axs[0], u_hebbRobot)
    # egoSphereHuman.plot(axs[1], h)
    # egoSphereRobot.plot(axs[0], networkRobot._W_obj[o,:], alpha=0.3)
    # egoSphereHuman.plot(axs[1], u_hebbRobot, alpha=0.3)

        k = 0
        for ax in axs :
            ax.view_init(elev=0, azim=1)
            if k == 0:                
                ax.view_init(elev=0, azim=0)
            else:
                ax.view_init(elev=0, azim=160)
                ax.invert_yaxis()
            k += 1
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_zlabel('z')
            ax.set_xlim([-0.75, 0.75])
            ax.set_ylim([-0.75, 0.75])
            ax.set_zlim([-0.01, 1.8])
            # ax.set_xticks([-0.75, 0.75])
            # ax.set_yticks([-0.75, 0.75])
            # ax.set_zticks([0.0, 1.8])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_aspect('equal')
            #ax.grid(False)
            ax.axis('off')

        with open('{}h_human.npy'.format(dataPath), 'wb') as f:
            np.save(f,np.vstack(all_h_human))
        with open('{}W_hr.npy'.format(dataPath), 'wb') as f:
            np.save(f,W_hr)
        with open('{}W_obj.npy'.format(dataPath), 'wb') as f:
            np.save(f,networkRobot._W_obj)

        plt.savefig('{}plot_{}.png'.format(dataPath, o))
        #plt.show()
    
print("program end")
    

