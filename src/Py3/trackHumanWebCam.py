import cv2
from AEGO.Human import HumanTracker
import mmap
import numpy as np
import posix_ipc as pos
import struct
import json
import sys 

def main(params):

    print("program start")

    file = params.get('videoFile')
    fps = params['imgFPSWeb']
    user = params['expID']
    pVersion = 3

    cap = None
    
    # For webcam input:
    if file is None :
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file)

    sleepInMS = int(1000/fps) 

    humanTracker = HumanTracker() 
    if cap is None:
        print('invalid device!\n program end')

     # landmark shared memory
    nJointValues = 33
    singlePrecisionInBytes = 4
    bJointSize = nJointValues * 4 * singlePrecisionInBytes
    mem = pos.SharedMemory('/{}_mediapipe'.format(user), pos.O_CREAT,size=bJointSize)
    memHuman = mmap.mmap(mem.fd, bJointSize)
    sizeMemHuman = mem.size
    print("memLandmark size in bytes: {}".format(sizeMemHuman))

    while (True):
        success, image = cap.read()
        if success:
            
            imgShape = (320, 240)
            resized = cv2.resize(image, imgShape, interpolation= cv2.INTER_LINEAR)
            
            img_seg, joints = humanTracker.step(resized)

            # sending human detection
            buf = []
            for j in joints.flatten().tolist():
                buf += list(struct.pack("f", j))
            memHuman.seek(0)
            
            if pVersion == "2":
                memHuman.write(str(bytearray(buf)))
            else:
                memHuman.write(bytearray(buf))
            memHuman.flush()

            cv2.imshow('MediaPipe Pose', img_seg)
            if cv2.waitKey(sleepInMS) & 0xFF == 27:
                cv2.destroyWindow('MediaPipe Pose')
                break

        else :
            break

    cap.release()
    print("program end")

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

    params['videoFile'] = '/home/hfchame/Videos/Webcam/2024-03-10-100814.webm'
    main(params)
