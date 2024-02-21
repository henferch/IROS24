#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

""" Main program to control the robot. Data is sent and received to other process via POSIX shared """

import qi
import argparse
import sys
import time
from JAEgoQi.HumanTrackService import HumanTrackService
from JAEgoQi.ObjectTrackService import ObjectTrackService
from JAEgoQi.SpeechRecognitionService import SpeechRecognitionService

class MonAppli(object):
    def __init__(self, app):
        super(MonAppli, self).__init__()
        app.start()
        session = app.session
        
        # service proxies
        self.motion_srv = session.service("ALMotion") 
        self.posture_srv = session.service("ALRobotPosture") 
        self.autLife_srv = session.service("ALAutonomousLife") 
        self.video_srv = session.service("ALVideoDevice") 
        self.memory_srv = session.service("ALMemory")
        self.landmark_srv = session.service("ALLandMarkDetection")
        self.speech_srv = session.service("ALSpeechRecognition")

        # go to Stant up posture
        # self.posture.goToPosture("Stand", 0.5)

        # disabling autonomous life
        self.setAutonomousLife(False)
        
        # setting up services
        self.humanTracker = HumanTrackService(self.video_srv, self.motion_srv)
        self.objectTracker = ObjectTrackService(self.memory_srv, self.landmark_srv, self.motion_srv)
        self.speechRecogn = SpeechRecognitionService(self.memory_srv, self.speech_srv)
           
    def setAutonomousLife(self, valeur=True):
        self.autLife_srv.setAutonomousAbilityEnabled("AutonomousBlinking", valeur)
        self.autLife_srv.setAutonomousAbilityEnabled("BackgroundMovement", valeur)
        self.autLife_srv.setAutonomousAbilityEnabled("BasicAwareness", valeur)
        self.autLife_srv.setAutonomousAbilityEnabled("ListeningMovement", valeur)
        self.autLife_srv.setAutonomousAbilityEnabled("SpeakingMovement", valeur)

    def stop(self):
        # 1) stop motion
        self.motion_srv.stopMove()
        #self.motion.rest()
        print("Robot in rest state")
        
        # 2) unsubscribe from services
        self.humanTracker.stop()
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
            self.humanTracker.step()
            # for k,v in self.objectTracker.getObjects().items():
            #     print(k,v)
            t2 = time.time()
            #print('loop time in ms : {:.3f}'.format((t2-t1)*1000))

                

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
