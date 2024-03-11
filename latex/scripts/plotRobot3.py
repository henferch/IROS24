#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Use transformInterpolations Method on Arm"""

import qi
import argparse
import sys
import motion
import math
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from Network import Network
from PlotTools import *
from GeodesicDome import GeodesicDome
from RobotPlot import RobotPlot
from Utils import Utils

def render(ax, robot, egoSph, rightArmEgo, leftArmEgo, network, ut):    
    
    robot.updateRobotFramesPos()
    robot.render()

    Torso, RElbowRoll, RWristYaw, LElbowRoll, LWristYaw = robot.getIntersectionPoints()

    pR1, pR2 = egoSph.intersect(RElbowRoll, RWristYaw-RElbowRoll)
    pL1, pL2 = egoSph.intersect(LElbowRoll, LWristYaw-LElbowRoll)

    network.updateObjects([pR1-Torso, pL1-Torso])
    u_pre, u_sel, o = network.step({'o':[1.0, 1.0], 'l' : 0.0, 'r': 0.0, 'a':0.0, 'b':0.0, 'n': 0.0})

    egoSph.render(u_pre)
    #egoSph.render(u_sel)
    
    # # plotting arm-ego intersection point 
    ut.setPlotData3DPoint(leftArmEgo, pL1)
    ut.setPlotData3DPoint(rightArmEgo, pR1)
    
    ax.figure.canvas.draw()
    
def main(session):
    """
    Use case of transformInterpolations API.
    """
    # Get the services ALMotion & ALRobotPosture.


    ut = Utils.getInstance()

    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")

    # Wake up robot
    #motion_service.wakeUp()

    # Send robot to Stand Init
    #posture_service.goToPosture("StandInit", 0.5)

    # print(motion_service.getBodyNames('Body'))
    # print(motion_service.getSensorNames())

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(10, 10))
    fig.tight_layout()
    ax.view_init(elev=0, azim=0)

    res = 16
    radius = 0.25

    robot = RobotPlot(motion_service, ut)
    center = robot.getBaseLocation()
    robot.plot(ax,'darkcyan', 3.0)
    
    # Icosahedron
    tesselation = 3        
    sigma = 0.001
    objects = [np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0])]
    params = {'tesselation': tesselation, 'scale' : radius, 'center': center}        
    egoSphere = GeodesicDome(params)

    refs = egoSphere.getV()
    dt = 0.05        
    params = {\
        'ut': ut,
        'ref': refs,
        'res' : res,
        'sig': np.eye(3)*sigma,
        'h_pre' : -0.01,
        'h_sel' : -0.0001,
        'dt' : dt,
        #'inh' : 0.0001,
        'inh' : 0.0001,
        'tau' : 0.2,
        'objects': objects
        }              

    network = Network(params) 
        
    # plotting a dummy intersection point        
    rightArmEgo = ut.plot3DPoint(ax, center, 'red')
    leftArmEgo = ut.plot3DPoint(ax, center, 'red')

    egoSphere.plot(ax, network.getU_pre())
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-0.75, 0.75])
    ax.set_ylim([-0.75, 0.75])
    ax.set_zlim([-0.01, 1.5])
    ax.set_xticks([-0.75, 0.75])
    ax.set_yticks([-0.75, 0.75])
    ax.set_zticks([0.0, 1.5])
    ax.set_aspect('equal')
    ax.grid(False)

    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(render, ax, robot, egoSphere, rightArmEgo, leftArmEgo, network, ut)
    timer.start()

    plt.show()
    
    print("plot end")
    timer.stop()
   
    #time.sleep(2.0)

    # Go to rest position
    #motion_service.rest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)