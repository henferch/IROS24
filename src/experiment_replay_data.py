import numpy as np
import cv2
import posix_ipc as pos
import json
import struct
import mmap
import sys
import math

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

I = [340, 387, 297, 247, 269, 257, 283]
F = [1676, 1455, 1317, 1267, 1109, 1104, 1344]

fps = 10
timeSleep = int(1000.0/fps)

# ExpData shared memory
nValues = 30 
singlePrecisionInBytes = 4
bJointSize = nValues * 3 * singlePrecisionInBytes
mem = pos.SharedMemory('/{}_expData'.format(user), pos.O_CREAT,size=bJointSize)
memExpData = mmap.mmap(mem.fd, bJointSize)
sizeMemExpData = mem.size
print("memPosture size in bytes: {}".format(sizeMemExpData))


stop = False
#for e in range(5):
for e in [6]:
    print("processisng experiment ".format(e))
    exp = ExpID[e]
    humanPose = None
    with open('/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/humanPose.npy'.format(exp), 'rb') as f:
        humanPose = np.load(f)

    position = None
    with open('/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/position.npy'.format(exp), 'rb') as f:
        position = np.load(f)

    robotPose = None
    with open('/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/robotPose.npy'.format(exp), 'rb') as f:
        robotPose = np.load(f)

    robotDemo = None
    with open('/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/robotDemo.npy'.format(exp), 'rb') as f:
        robotDemo = np.load(f)
    
    humanDemo = None
    with open('/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/humanDemo.npy'.format(exp), 'rb') as f:
        humanDemo = np.load(f)

    #print(position.shape)
    #humanPose.reshape(robotPose.shape)
    robotPose = robotPose.reshape((humanPose.shape[0], 14, 3))
    humanDemo = humanDemo.reshape((robotDemo.shape[0], 16, 3))
    print('exp', exp)  

    print(humanPose.shape)
    print(robotPose.shape)
    print(humanDemo.shape)
    print(robotDemo.shape)
    print(F[e]-I[e], (F[e]-I[e])/humanPose.shape[0])
    
    ftime = np.linspace(0,humanPose.shape[0]-1, F[e]-I[e]+1)
    
    t = 0
    for i in range(I[e], F[e]+1):
        img_url = '/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/frames/img_{}.png'.format(exp,i)
        seg_url = '/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/frames/seg_{}.png'.format(exp,i)

        #print(img_url)        
        img = cv2.imread(img_url)    
        seg = cv2.imread(seg_url)

        cv2.imshow('Img EXP{}'.format(exp), img)
        cv2.imshow('Seg EXP{}'.format(exp), seg)
        
        t_i = math.floor(ftime[t])
        t += 1
        human = humanPose[t_i,:].tolist()
        robot = robotPose[t_i,:].tolist()
        # sending posture data
        buf = []
        for p in robot+human :
            for c in p :
                buf += list(struct.pack("f", c))
        
        memExpData.seek(0)
        #memExpData.write(str(bytearray(buf))) # for python 2
        memExpData.write(bytearray(buf))
        memExpData.flush()

        if cv2.waitKey(timeSleep) & 0xFF == 27:
            cv2.destroyWindow('Img EXP{}'.format(exp))
            cv2.destroyWindow('Seg EXP{}'.format(exp))
            stop = True
            break
    if stop:
        break
    else:
        cv2.destroyWindow('Img EXP{}'.format(exp))
        cv2.destroyWindow('Seg EXP{}'.format(exp))
print("program end")
    