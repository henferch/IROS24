#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Use transformInterpolations Method on Arm"""

import qi
import argparse
import sys
import motion
import almath
import math
import time
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
	
def intersect(r_origin, r_dir, s_center, s_radius):
    P1 = None
    P2 = None
    #solve for tc
    L = s_center - r_origin
    tc = np.dot(L, r_dir)
    if (tc >= 0.0):
        d2 = (tc*tc) - (np.dot(L,L))
        radius2 = s_radius * s_radius
        if (d2 <= radius2):
            #solve for t1c
            t1c = np.sqrt( radius2 - d2 )
            #solve for intersection points
            t1 = tc - t1c
            t2 = tc + t1c
            P1 = r_origin + r_dir * t1
            P2 = r_origin + r_dir * t2 
    return P1, P2

# unfortunately there is o simple way to get all transforms at once
def getTransfrom(motion_service, name):
    frame  = motion.FRAME_ROBOT #FRAME_TORSO
    useSensorValues  = True
    f = motion_service.getTransform(name, frame, useSensorValues)
    return np.array(f).reshape((4,4))

def getRobotFramesPos(motion_service):
    
    dFrames = {} 
    dFrames['RShoulderRoll'] = getTransfrom(motion_service,'RShoulderRoll')[:,3] 
    dFrames['RElbowRoll'] = getTransfrom(motion_service,'RElbowRoll')[:,3] 
    dFrames['RWristYaw'] = getTransfrom(motion_service,'RWristYaw')[:,3] 
    dFrames['RArm'] = getTransfrom(motion_service,'RArm')[:,3]
    dFrames['LShoulderRoll'] = getTransfrom(motion_service,'LShoulderRoll')[:,3] 
    dFrames['LElbowRoll'] = getTransfrom(motion_service,'LElbowRoll')[:,3] 
    dFrames['LWristYaw'] = getTransfrom(motion_service,'LWristYaw')[:,3] 
    dFrames['LArm'] = getTransfrom(motion_service,'LArm')[:,3]
    dFrames['HeadYaw'] = getTransfrom(motion_service,'HeadYaw')[:,3]
    dFrames['HeadTouch'] = getTransfrom(motion_service,'Head/Touch/Front')[:,3]
    dFrames['HipPitch'] = getTransfrom(motion_service,'HipPitch')[:,3] 
    dFrames['KneePitch'] = getTransfrom(motion_service,'KneePitch')[:,3] 
    dFrames['Leg'] = getTransfrom(motion_service,'Leg')[:,3]
    return dFrames


def plot3DLine(ax, P1, P2, color):
    return ax.plot3D([P1[0],P2[0]], [P1[1],P2[1]], [P1[2],P2[2]], color)[0]
    
def plotRobotBody(motion_service, ax):
    
    dFrames = getRobotFramesPos(motion_service)

    #plotting head 
    shoulder_center = (dFrames['RShoulderRoll'] + dFrames['LShoulderRoll']) / 2.0
    o_Shoulder_Neck = plot3DLine(ax, shoulder_center, dFrames['HeadYaw'], 'blue')
    o_Neck_HeadTouch = plot3DLine(ax, dFrames['HeadYaw'], dFrames['HeadTouch'], 'blue')

    #plotting arms
    o_RShoulder_RElbow = plot3DLine(ax, dFrames['RShoulderRoll'], dFrames['RElbowRoll'], 'blue')
    o_RElbow_RWrist = plot3DLine(ax, dFrames['RElbowRoll'], dFrames['RWristYaw'], 'blue')
    o_Rwrist_RHand = plot3DLine(ax, dFrames['RWristYaw'], dFrames['RArm'], 'blue')
    o_LShoulder_LElbow = plot3DLine(ax, dFrames['LShoulderRoll'], dFrames['LElbowRoll'], 'blue')
    o_LElbow_LWrist = plot3DLine(ax, dFrames['LElbowRoll'], dFrames['LWristYaw'], 'blue')
    o_LWrist_LHand = plot3DLine(ax, dFrames['LWristYaw'], dFrames['LArm'], 'blue')
    o_RShoulder_LShoulder = plot3DLine(ax, dFrames['RShoulderRoll'], dFrames['LShoulderRoll'], 'blue')
    
    #plotting low extremity
    o_Shoulder_HipPitch = plot3DLine(ax, shoulder_center, dFrames['HipPitch'], 'blue')
    o_HipPitch_KneePitch = plot3DLine(ax, dFrames['HipPitch'], dFrames['KneePitch'], 'blue')
    o_KneePitch_Leg = plot3DLine(ax, dFrames['KneePitch'], dFrames['Leg'], 'blue')
    
    objects = {}
    objects['RShoulder_LShoulder'] = o_RShoulder_LShoulder
    objects['RShoulder_RElbow'] = o_RShoulder_RElbow
    objects['RElbow_RWrist'] = o_RElbow_RWrist
    objects['RWrist_RHand'] = o_Rwrist_RHand
    objects['LShoulder_LElbow'] = o_LShoulder_LElbow
    objects['LElbow_LWrist'] = o_LElbow_LWrist
    objects['LWrist_LHand'] = o_LWrist_LHand
    objects['Shoulder_Neck'] = o_Shoulder_Neck
    objects['Neck_HeadTouch'] = o_Neck_HeadTouch
    objects['Shoulder_HipPitch'] = o_Shoulder_HipPitch
    objects['HipPitch_KneePitch'] = o_HipPitch_KneePitch
    objects['KneePitch_Leg'] = o_KneePitch_Leg

    return objects
    

def plotEgoSphere(ax, center):
    theta, phi = np.linspace(0, 2 * np.pi, 12), np.linspace(0, np.pi, 12)
    THETA, PHI = np.meshgrid(theta, phi)
    R = 0.25
    X = R * np.sin(PHI) * np.cos(THETA) + center[0]
    Y = R * np.sin(PHI) * np.sin(THETA) + center[1]
    Z = R * np.cos(PHI) + center[2]
    plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, color='red', antialiased=False, alpha=0.3)

def set_data_3DLine(obj, P1, P2):
    obj.set_data([P1[0],P2[0]], [P1[1],P2[1]])
    obj.set_3d_properties([P1[2],P2[2]])

def render(motion_service, ax, objects):
    
    dFrames = getRobotFramesPos(motion_service)

    #plotting head 
    shoulder_center = (dFrames['RShoulderRoll'] + dFrames['LShoulderRoll']) / 2.0
    set_data_3DLine(objects['Shoulder_Neck'], shoulder_center, dFrames['HeadYaw'])
    set_data_3DLine(objects['Neck_HeadTouch'], dFrames['HeadYaw'], dFrames['HeadTouch'])

    #plotting arms
    set_data_3DLine(objects['RShoulder_RElbow'],dFrames['RShoulderRoll'], dFrames['RElbowRoll'])
    set_data_3DLine(objects['RElbow_RWrist'], dFrames['RElbowRoll'], dFrames['RWristYaw'])
    set_data_3DLine(objects['RWrist_RHand'], dFrames['RWristYaw'], dFrames['RArm'])
    set_data_3DLine(objects['LShoulder_LElbow'], dFrames['LShoulderRoll'], dFrames['LElbowRoll'])
    set_data_3DLine(objects['LElbow_LWrist'], dFrames['LElbowRoll'], dFrames['LWristYaw'])
    set_data_3DLine(objects['LWrist_LHand'], dFrames['LWristYaw'], dFrames['LArm'])
    set_data_3DLine(objects['RShoulder_LShoulder'], dFrames['RShoulderRoll'], dFrames['LShoulderRoll'])
    
    #plotting low extremity
    set_data_3DLine(objects['Shoulder_HipPitch'], shoulder_center, dFrames['HipPitch'])
    set_data_3DLine(objects['HipPitch_KneePitch'], dFrames['HipPitch'], dFrames['KneePitch'])
    set_data_3DLine(objects['KneePitch_Leg'], dFrames['KneePitch'], dFrames['Leg'])

    ax.figure.canvas.draw()
    print('render')

def main(session):
    """
    Use case of transformInterpolations API.
    """
    # Get the services ALMotion & ALRobotPosture.

    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")

    # Wake up robot
    #motion_service.wakeUp()

    # Send robot to Stand Init
    #posture_service.goToPosture("StandInit", 0.5)

    print(motion_service.getBodyNames('Body'))
    print(motion_service.getSensorNames())
    
    torso = getTransfrom(motion_service,'Torso')[:,3]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    objs = plotRobotBody(motion_service, ax)

    plotEgoSphere(ax, torso)
    
    ax.set_xlim([-0.75, 0.75])
    ax.set_ylim([-0.75, 0.75])
    ax.set_zlim([-0.01, 1.5])
    ax.set_xticks([-0.75, 0.75])
    ax.set_yticks([-0.75, 0.75])
    ax.set_zticks([0.0, 1.5])
    ax.set_aspect('equal')
    ax.grid(False)

    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(render, motion_service, ax, objs)
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