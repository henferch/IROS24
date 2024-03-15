#! /usr/bin/env python3
# -*- encoding: UTF-8 -*-

"""Example: Get an image. Display it and save it using PIL."""

import sys
import time
import numpy as np
import cv2
import mmap
import posix_ipc as pos
import struct
import json
from AEGO.Human import HumanTracker


def main(params):
    user = params['expID']
    record = params['camTrackerSaveData'] == "yes"
    fps = params['imgFPSWeb']
    sleepInMS = int(1000/fps) 

    allTimes = []
    allFrames = []
    allFramesSeg = []
    allPositions = []
    humanTracker = HumanTracker()  
    pVersion = "3"

    # Image size VGA
    #imgShape = (480,640, 3)
    imgShape = (240,320, 3)
    imgSize = int(np.prod(imgShape))
    
    # Shared memory 
    mem = pos.SharedMemory('/{}_image'.format(user), pos.O_CREAT,size=imgSize)
    mm = mmap.mmap(mem.fd, imgSize)
    print("memory size in bytes: {}".format(mem.size))

    # landmark shared memory
    nJointValues = 33
    singlePrecisionInBytes = 4
    bJointSize = nJointValues * 4 * singlePrecisionInBytes
    mem2 = pos.SharedMemory('/{}_mediapipe'.format(user), pos.O_CREAT,size=bJointSize)
    memHuman = mmap.mmap(mem2.fd, bJointSize)
    sizeMemHuman = mem2.size
    print("memLandmark size in bytes: {}".format(sizeMemHuman))
    
    while (True):
        t1 = time.time()
        # getting image
        imgData = np.zeros((imgSize,), dtype=np.ubyte)
        if pVersion == "2":
            for i in range(imgSize):
                imgData[i] = ord(mm[i])
        else: 
            for i in range(imgSize):
                imgData[i] = mm[i]
        
        img = imgData.reshape(imgShape)    
        img_RGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_seg, joints = humanTracker.step(img_RGB)

        ##print(joints)

        # sending human detection
        buf = []
        for j in joints.flatten().tolist():
            buf += list(struct.pack("f", j))
        memHuman.seek(0)
        
        if pVersion[0] == "2":
            memHuman.write(str(bytearray(buf)))
        else:
            memHuman.write(bytearray(buf))
        memHuman.flush()

        cv2.imshow('MediaPipe Pose', img_seg)
        t2 = time.time()
        if record:
            allFrames.append(img_RGB)
            allFramesSeg.append(img_seg)
            allPositions.append(joints)
            allTimes.append(t2 - t1)

        if cv2.waitKey(sleepInMS) & 0xFF == 27:
            cv2.destroyWindow('MediaPipe Pose')
            break

    if record:
        k = 0
        for i, s in zip (allFrames, allFramesSeg):
            cv2.imwrite("../data/img_{}.png".format(k), i)
            cv2.imwrite("../data/seg_{}.png".format(k), s)
            k += 1

        with open('../data/position.npy', 'wb') as f:
            np.save(f, np.vstack(allPositions))
        with open('../data/times.npy', 'wb') as f:
            np.save(f, np.array(allTimes))


if __name__ == "__main__":
      # Opening JSON file
    params = None
    try:
        f = open('/home/hfchame/Workspace/VSCode/IROS24/src/parameters.json')
        params = json.load(f)
        f.close()
    except Exception as ex:
        print ("Can't load parameters from file 'parameters.json'. Error: {}".format(ex))
        sys.exit(1)

    main(params)
