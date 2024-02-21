#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

""" Main program to control the robot. Data is sent and received to other process via POSIX shared """

import qi
import argparse
import sys
import time
import json
import struct
import mmap
import posix_ipc as pos
from JAEgoQi.PostureTrackService import PostureTrackService
from JAEgoQi.ObjectTrackService import ObjectTrackService
from JAEgoQi.SpeechRecognitionService import SpeechRecognitionService

class MonAppli(object):
    def __init__(self, app):
        super(MonAppli, self).__init__()
        app.start()

        self.user = "ManipHFC2024" 

        # Opening JSON file
        f = open('parameters.json')
        self.parameters = json.load(f)
        f.close()

        session = app.session
        
        # service proxies
        self.motion_srv = session.service("ALMotion") 
        self.posture_srv = session.service("ALRobotPosture") 
        self.autLife_srv = session.service("ALAutonomousLife")
        self.basicAwareness_srv = session.service("ALBasicAwareness")
        self.faceDetection_srv = session.service("ALFaceDetection") 
        self.video_srv = session.service("ALVideoDevice") 
        self.memory_srv = session.service("ALMemory")
        self.landmark_srv = session.service("ALLandMarkDetection")
        self.speech_srv = session.service("ALSpeechRecognition")

        # disabling autonomous life
        self.setAutonomousLife(False)
        
        # go to Stant up posture
        self.posture_srv.goToPosture("Stand", 0.2)
        time.sleep(1)
        fractionMaxSpeed  = 0.2
        self.motion_srv.setAngles(['HeadPitch'], [0.02], fractionMaxSpeed)
        time.sleep(1)
        
        # ExpData shared memory
        nValues = 30 
        singlePrecisionInBytes = 4
        bJointSize = nValues * 3 * singlePrecisionInBytes
        mem = pos.SharedMemory('/{}_expData'.format(self.user), pos.O_CREAT,size=bJointSize)
        self.memExpData = mmap.mmap(mem.fd, bJointSize)
        self.sizeMemExpData = mem.size
        print("memPosture size in bytes: {}".format(self.sizeMemExpData))

        self.postureTracker = PostureTrackService(self.video_srv, self.motion_srv, self.parameters)
        self.objectTracker = ObjectTrackService(self.memory_srv, self.landmark_srv, self.motion_srv, self.parameters)
        self.speechRecogn = SpeechRecognitionService(self.memory_srv, self.speech_srv, self.parameters)
           
    def setAutonomousLife(self, valeur=True):

        # self.basicAwareness_srv.pauseAwareness()
        # self.faceDetection_srv.setTrackingEnabled(False)
        
        self.autLife_srv.setAutonomousAbilityEnabled("AutonomousBlinking", valeur)
        self.autLife_srv.setAutonomousAbilityEnabled("BackgroundMovement", False)
        self.autLife_srv.setAutonomousAbilityEnabled("BasicAwareness", False)
        self.autLife_srv.setAutonomousAbilityEnabled("ListeningMovement", False)
        self.autLife_srv.setAutonomousAbilityEnabled("SpeakingMovement", False)

    def stop(self):
        # 1) stop motion
        self.motion_srv.stopMove()
        #self.motion.rest()
        print("Robot in rest state")
        
        # 2) unsubscribe from services
        self.postureTracker.stop()
        self.objectTracker.stop()
        self.speechRecogn.stop()
        
        # 3) re-active AL
        self.setAutonomousLife(True)

        # 4) Program end
        print("Program end")
        sys.exit(1)

    def run(self):
        while (True):
            t1 = time.time()
            robot, human = self.postureTracker.step()
            objects = self.objectTracker.getObjects()

            # sending posture data
            buf = []
            for p in robot+human :
                for c in p :
                    buf += list(struct.pack("f", c))

            self.memExpData.seek(0)
            self.memExpData.write(str(bytearray(buf)))
            self.memExpData.flush()
            
            t2 = time.time()
            print('loop time in ms : {:.3f}'.format((t2-t1)*1000))

                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    robot_IP = "127.0.0.1"
    robot_IP = "192.168.137.166"

    parser.add_argument("--ip", type=str, default=robot_IP,
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()

    print("Debut d'application")    
    try:
        # Initialize qi framework.
        connection_url = "tcp://" + args.ip + ":" + str(args.port)
        app = qi.Application(["MyApplicaction", "--qi-url=" + connection_url])
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    monAppli = MonAppli(app)
    try:
        monAppli.run()
    except KeyboardInterrupt:
        print("Programme arrêté par l'utilisateur")
        monAppli.stop()
    except RuntimeError as err:
        print("Erreur d'éxecution : ", err)
        monAppli.stop()
