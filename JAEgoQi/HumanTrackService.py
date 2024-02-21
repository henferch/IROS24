import numpy as np
import mmap
import posix_ipc as pos
import vision_definitions
import almath
import motion
from Service import Service

class HumanTrackService(Service):
    def __init__(self, video_srv, motion_srv):
        Service.__init__(self)
        self.video_srv = video_srv
        self.motion_srv = motion_srv

        # Image size VGA
        #imgShape = (480,640, 3)
        imgShape = (240,320, 3)
        imgSize = np.prod(imgShape)
        
        resolution = vision_definitions.kQVGA
        colorSpace = vision_definitions.kRGBColorSpace
        fps = 10
        camID = 0 # top camera

        subscribers = self.video_srv.getSubscribers()
        print("subscribers (before): ", subscribers )

        for s in subscribers:
            if self.user in s:
                self.video_srv.unsubscribe(s)

        # subscribe to video client
        self.videoClient = self.video_srv.subscribeCamera(self.user, camID, resolution, colorSpace, fps)

        # image shared memory
        mem1 = pos.SharedMemory('/{}_image'.format(self.user), pos.O_CREAT,size=imgSize)
        self.memImage = mmap.mmap(mem1.fd, imgSize)
        self.sizeMemImage = mem1.size
        print("memImage size in bytes: {}".format(self.sizeMemImage))

        # landmark shared memory
        nJointValues = 33
        singlePrecisionInBytes = 4
        bJointSize = nJointValues * 4 * singlePrecisionInBytes
        mem2 = pos.SharedMemory('/{}_mediapipe'.format(self.user), pos.O_CREAT,size=bJointSize)
        self.memHuman = mmap.mmap(mem2.fd, bJointSize)
        self.sizeMemHuman = mem2.size
        print("memLandmark size in bytes: {}".format(self.sizeMemHuman))

        self.currentCamera = "CameraTop"

    def stop(self):
        for s in self.video_srv.getSubscribers():
            if self.user in s:
                self.video_srv.unsubscribe(s)
                
    def step(self):
        #t0 = time.time()
        # Get a camera image from the robot.
        robotImage = self.video_srv.getImageRemote(self.videoClient)
        array = robotImage[6]
        
        # send cam image
        self.memImage.seek(0)
        for i in range (self.sizeMemImage):
            self.memImage[i] = chr(array[i])
        
        # get detected human
        self.memHuman.seek(0)
        bufJoint = self.memHuman.read(self.sizeMemHuman)
        jList = np.frombuffer(bufJoint, dtype=np.float32).reshape((33,4))

        frame  = motion.FRAME_ROBOT #FRAME_TORSO
        transform = self.motion_srv.getTransform(self.currentCamera, frame, True)
        robotToCamera = np.array(transform).reshape(4,4)
        pInRobotFrame = [] 
        #print('robotToCam', robotToCamera)
        for i in range (13,33):
            jList_i = jList[i,:]
            if jList_i[3] > 0.8:
                #mediapipe XYZ--> Pepper (-Z)X,Y
                cameraPoint = np.array([-jList_i[2], jList_i[0], jList_i[1], 1.0]).reshape(4,)
                p = np.matmul(robotToCamera, cameraPoint) 
                pInRobotFrame.append([i, [p[2],p[0],p[1]]])

        self.motion_srv
        # print(jList)
        # print(pInRobotFrame)
        