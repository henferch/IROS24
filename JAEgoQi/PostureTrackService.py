import numpy as np
import mmap
import posix_ipc as pos
import vision_definitions
import almath
import motion
from Service import Service
from operator import itemgetter


""" This class implements all the requirements for tracking both the human and robot posture """
class PostureTrackService(Service):
    def __init__(self, video_srv, motion_srv, params={}):
        Service.__init__(self)
        self.video_srv = video_srv
        self.motion_srv = motion_srv

        self.humanDepthInMeter = params['humanDepthInMeter']

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

        self.cameraId = "CameraTop"

        self.robotPoints = []
        self.humanPoints = []
        self.frame = motion.FRAME_ROBOT

    def stop(self):
        for s in self.video_srv.getSubscribers():
            if self.user in s:
                self.video_srv.unsubscribe(s)
                
    def step(self):

        # Getting the robot body points
        # this is not the best way to do this, since the joint positions should
        # be queried at once. The problem is that Pepper does not have plenty of ressources
        # implemented, and there is no time to rebuilt de Deravit-Hatternberg parameters   
        self.robotPoints = [itemgetter(3,7,11)(self.motion_srv.getTransform('RShoulderRoll',    self.frame, True)),\
                            itemgetter(3,7,11)(self.motion_srv.getTransform('RElbowRoll',       self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('RWristYaw',        self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('RArm',             self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('LShoulderRoll',    self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('LElbowRoll',       self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('LWristYaw',        self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('LArm',             self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('HeadYaw',          self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('Head/Touch/Front', self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('HipPitch',         self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('KneePitch',        self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('Leg',              self.frame, True)),
                            itemgetter(3,7,11)(self.motion_srv.getTransform('Torso',            self.frame, True))]

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

        transform = self.motion_srv.getTransform(self.cameraId, self.frame, True)
        robotToCamera = np.array(transform).reshape(4,4)
        
        self.humanPoints = [] 
        # expressing the human points in RobotFrame

        for i in [4, 12, 14, 16, 20, 24, 26, 28, 1, 11, 13, 15, 19, 23, 25, 27]:
            jList_i = jList[i,:]
            #mediapipe XYZ--> Pepper (-Z)X,Y
            #cameraPoint = np.array([jList_i[2]+self.humanDepthInMeter, -jList_i[0], -jList_i[1], 1.0]).reshape(4,)
            #p = np.matmul(robotToCamera, cameraPoint)
            #self.humanPoints.append(p[0:3].tolist()) 
            self.humanPoints.append([jList_i[2], jList_i[0], -jList_i[1]])
        return self.robotPoints, self.humanPoints
        