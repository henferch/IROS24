#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Ce qui fait cette application d'application :
	a) Désactivation du module de vie autonome
	b) Le robot est mis debout
	c) Boucle infinie qui attend la fin d'exécution par via CONTROL+C
	d) Remise du robot dans l'état de repos
 """

import qi
import argparse
import sys
import time
from HumanTrackService import HumanTrackService
from ObjectTrackService import ObjectTrackService

class MonAppli(object):
    def __init__(self, app):
        super(MonAppli, self).__init__()
        app.start()
        session = app.session
        
        # Proxies aux services
        self.motion_srv = session.service("ALMotion") # gestion du mouvement.
        self.posture_srv = session.service("ALRobotPosture") # Gestion de la posture.
        self.autLife_srv = session.service("ALAutonomousLife") # Gestion de la vie autonome.
        self.video_srv = session.service("ALVideoDevice") # Gestion video.
        self.memory_srv = session.service("ALMemory")
        self.landmark_srv = session.service("ALLandMarkDetection")

        self.humanTracker = HumanTrackService(self.video_srv, self.motion_srv)
        self.objectTracker = ObjectTrackService(self.memory_srv, self.landmark_srv, self.motion_srv)
        
        # Aller debout 
        #self.posture.goToPosture("Stand", 0.5)

        # désactiver la vie autonome
        self.activationVieAuto(False)

    def activationVieAuto(self, valeur=True):
        # Desactiver la vie autonome (si besoin)
        self.autLife_srv.setAutonomousAbilityEnabled("AutonomousBlinking", valeur)
        self.autLife_srv.setAutonomousAbilityEnabled("BackgroundMovement", valeur)
        self.autLife_srv.setAutonomousAbilityEnabled("BasicAwareness", valeur)
        self.autLife_srv.setAutonomousAbilityEnabled("ListeningMovement", False)
        self.autLife_srv.setAutonomousAbilityEnabled("SpeakingMovement", valeur)


    def stop(self):
        # 1) arreter et envoyer le robot à l'état de repos
        self.motion_srv.stopMove()
        #self.motion.rest()
        print("Robot mis en état de repos")
        
        # 2) se désinscrire de services si besoin
        # code ...
        self.humanTracker.stop()
        self.objectTracker.stop()

        self.activationVieAuto(True)

        # 3) finir l'execution  
        print("Fin d'application")
        sys.exit(1)

    def run(self):
        while (True):
            t1 = time.time()
            self.humanTracker.step()
            for k,v in self.objectTracker.getObjects().items():
                print(k,v)
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
