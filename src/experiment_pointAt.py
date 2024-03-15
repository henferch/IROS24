#! /usr/bin/env python
# -*- encoding: UTF-8 -*-


import numpy as np
import qi
import argparse
import sys
import time
import json
import threading
from JAEgoQi.PostureTrackService import PostureTrackService

class MonAppli(object):
    def __init__(self, app, params):
        super(MonAppli, self).__init__()
        app.start()

        self.parameters = params

        self.objects = {'64L': np.array([1.19082725, 0.45242429, 1.05075765]),\
                        '80L': np.array([ 1.91267562, -0.73908919,  1.09514868]), 
                        '68L': np.array([ 1.27697361, -0.09755165,  1.06898379]), 
                        '84L': np.array([1.67314959, 0.58602697, 1.53109407])}#, 
                        #'113L': np.array([ 1.97443807, -0.7649287 ,  1.07176149])}

        session = app.session
        
        # service proxies
        self.motion_srv = session.service("ALMotion") 
        self.video_srv = session.service("ALVideoDevice") 
        self.posture_srv = session.service("ALRobotPosture")
        self.autLife_srv = session.service("ALAutonomousLife")
        self.basicAwareness_srv = session.service("ALBasicAwareness")
        #self.speech_srv = session.service("ALSpeechRecognition")
        self.textToSpeech_srv = session.service("ALTextToSpeech")
        self.textToSpeech_srv.setLanguage("English")

        self.postureTracker = PostureTrackService(self.video_srv, self.motion_srv, self.parameters)

        self.tracker_srv = None 
        
        self.tracker_srv = session.service("ALTracker") 
        self.tracker_srv.setMode("WholeBody")
    
        # disabling autonomous life
        self.setAutonomousLife(False)
        
        # go to Stant up posture
        self.posture_srv.goToPosture("Stand", 0.2)
        time.sleep(1)
        fractionMaxSpeed  = 0.1
        self.motion_srv.setAngles(['HeadPitch'], [0.02], fractionMaxSpeed)
        time.sleep(1)
                
    def pointAt(self, arm, p, frame, s, frameHuman, frameRobot):
        self.tracker_srv.pointAt(arm, p, frame, 0.1)
        self.textToSpeech_srv.say('this is {}'.format(s))
        time.sleep(1)
        robot, human = self.postureTracker.step()
        frameHuman.append(human)
        frameRobot.append(robot)
        #self.posture_srv.goToPosture("Stand", 0.1)
        fractionMaxSpeed  = 0.1
        names = ["LElbowRoll", "LElbowYaw", "LHand", "LShoulderPitch", "LShoulderRoll", "LWristYaw", "RElbowRoll", "RElbowYaw", "RHand", "RShoulderPitch", "RShoulderRoll", "RWristYaw"]
        angle = [-0.302099, -1.4981, 0.450353, 1.66157, 0.0808528, -0.247417, 0.262406, 1.62856, 0.438848, 1.66012, -0.0626706, 0.2064]
        self.motion_srv.setAngles(names, angle, fractionMaxSpeed)
        #self.motion_srv.setAngles(['HeadPitch'], [0.1], fractionMaxSpeed)
        #self.motion_srv.setAngles(['LShoulderPitch', 'RShoulderPitch'], [np.pi/2.0, np.pi/2.0], fractionMaxSpeed)
        time.sleep(1)


    def run(self):
        
        FRAME_TORSO = 0
        FRAME_WORLD = 1
        FRAME_ROBOT = 2
        arms = ["Arms", "LArm", "RArm"]
        say = ['one', 'two', 'three', 'four']
        oKeys  = ['68L', '84L', '64L', '80L']

        s = 0
        allRobot = []
        allHuman = []
        allRobotDemo = []
        allHumanDemo = []

        time.sleep(5)
        self.textToSpeech_srv.say("come on!")
        self.textToSpeech_srv.say("Let's point with me !")
        
        t = 0.0
        dt = 0.1
        T= 20.0
        o = 0
        nextPointing = 0.0
        dtPointing = 4.0
        while(t < T):
        
            if t >= nextPointing and o < 4:
                p = self.objects[oKeys[o]].tolist()
                arm = arms[1]
                if p[1] < 0.0:
                    arm = arms[2]
                #self.pointAt(arm,p, FRAME_WORLD, say[o])
                x = threading.Thread(target=self.pointAt, args=(arm,p, FRAME_WORLD, say[o], allRobotDemo, allHumanDemo))
                x.start()
                nextPointing += dtPointing
                o += 1
            robot, human = self.postureTracker.step()
            allRobot.append(robot)
            allHuman.append(human)
            t += dt


            time.sleep(dt)

        # for k, v in self.objects.items():
        #     print(k)
        #     p = v.tolist()
        #     arm = arms[1]
        #     if p[1] < 0.0:
        #         arm = arms[2]
        #     self.tracker_srv.pointAt(arm, p, FRAME_WORLD, 0.1)
        #     self.textToSpeech_srv.say('this is {}'.format(say[s]))
        #     robot, human = self.postureTracker.step()
        #     allRobot.append(robot)
        #     allHuman.append(human)
            
        #     s += 1
        #     time.sleep(1)
        #     self.posture_srv.goToPosture("Stand", 0.1)
        #     fractionMaxSpeed  = 0.1
        #     self.motion_srv.setAngles(['HeadPitch'], [0.02], fractionMaxSpeed)
        #     #time.sleep(2)
        
        # with open('data/humanPose.npy', 'wb') as f:
        #     np.save(f, np.vstack(allRobot))
        # with open('data/robotPose.npy', 'wb') as f:
        #     np.save(f, np.array(allHuman))
        # with open('data/humanDemo.npy', 'wb') as f:
        #     np.save(f, np.vstack(allRobotDemo))
        # with open('data/robotDemo.npy', 'wb') as f:
        #     np.save(f, np.array(allHumanDemo))
        
        with open('data/humanPose.npy', 'wb') as f:
            np.save(f, np.array(allHuman))
            #np.save(f, np.vstack(allRobot))
        with open('data/robotPose.npy', 'wb') as f:
            np.save(f, np.array(allRobot))
            #np.save(f, np.array(allHuman))
        with open('data/humanDemo.npy', 'wb') as f:
            np.save(f, np.array(allHumanDemo))
            #np.save(f, np.vstack(allRobotDemo))
        with open('data/robotDemo.npy', 'wb') as f:
            np.save(f, np.array(allRobotDemo))
        

        
        self.stop()


    def setAutonomousLife(self, valeur=True):

        # self.basicAwareness_srv.pauseAwareness()
        # self.faceDetection_srv.setTrackingEnabled(False)
        
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
        self.postureTracker.stop()
            
        # 3) re-active AL
        self.setAutonomousLife(True)

        # 4) Program end
        print("Program end")
        sys.exit(1)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    robot_IP = "127.0.0.1"
    robot_IP = "192.168.137.166"

    parser.add_argument("--ip", type=str, default=robot_IP,
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()

    # Opening JSON file
    params = None
    try:
        f = open('/home/hfchame/Workspace/VSCode/IROS24/src/parameters.json')
        params = json.load(f)
        f.close()
    except Exception as ex:
        print ("Can't load parameters from file 'parameters.json'. Error: {}".format(ex))
        sys.exit(1)

    if args.ip == "127.0.0.1":
        params['robotSource'] = 'sim'
    else:
        params['robotSource'] = 'robot'

    print("Debut d'application")    
    try:
        # Initialize qi framework.
        connection_url = "tcp://" + args.ip + ":" + str(args.port)
        app = qi.Application(["MyApplicaction", "--qi-url=" + connection_url])
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    monAppli = MonAppli(app, params)
    try:
        monAppli.run()
    except KeyboardInterrupt:
        print("Programme arrêté par l'utilisateur")
        monAppli.stop()
    except RuntimeError as err:
        print("Erreur d'éxecution : ", err)
        monAppli.stop()
