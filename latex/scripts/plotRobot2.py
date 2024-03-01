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
from Network import Network
from PlotTools import *

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
    dFrames['Torso'] = getTransfrom(motion_service,'Torso')[:,3]
    return dFrames


def plot3DLine(ax, P1, P2, color, linewidth):
    return ax.plot3D([P1[0],P2[0]], [P1[1],P2[1]], [P1[2],P2[2]], color=color, linewidth=linewidth)[0]

def plot3DPoint(ax, P1, color):    
    #graph, = ax.plot3D(P1[0], P1[1], P1[2], color=color, marker='o', markersize=5.0)
    #return ax.scatter(P1[0], P1[1], P1[2], color=color, marker='o')
    return ax.plot([P1[0]], [P1[1]], [P1[2]], color=color, marker='o')[0]    

def set_data_3DLine(obj, P1, P2):
    obj.set_data([P1[0],P2[0]], [P1[1],P2[1]])
    obj.set_3d_properties([P1[2],P2[2]])

def set_data_3DPoint(obj, P1):
    obj.set_data(P1[0], P1[1])
    obj.set_3d_properties(P1[2])
    #obj._offsets3d = (P1[0], P1[1], P1[2])
    #obj.set_data (float(P1[0]), float(P1[1]), float(P1[3]))

def intersect(s_center, s_radius, r_origin, r_dir):
        P1 = None
        P2 = None
        o = r_origin
        u = r_dir / np.linalg.norm(r_dir)
        c = s_center
        r = s_radius
        o_c = o-c
        dot_u_o_c = np.dot(u,o_c)
        delta = dot_u_o_c**2.0 - (np.linalg.norm(o_c)**2.0 - r**2.0)
        if delta > 0:
            delta_sqrt = delta**0.5
            d1 = -dot_u_o_c + delta_sqrt  
            P1 = o + u*d1
            d2 = -dot_u_o_c - delta_sqrt
            P2 = o + u*d2
        return P1,P2

def plotRobotBody(motion_service, ax, color, linewidth):
    
    dFrames = getRobotFramesPos(motion_service)

    #plotting head 
    shoulder_center = (dFrames['RShoulderRoll'] + dFrames['LShoulderRoll']) / 2.0
    o_Shoulder_Neck = plot3DLine(ax, shoulder_center, dFrames['HeadYaw'], color, linewidth)
    o_Neck_HeadTouch = plot3DLine(ax, dFrames['HeadYaw'], dFrames['HeadTouch'], color, linewidth)

    #plotting arms
    o_RShoulder_RElbow = plot3DLine(ax, dFrames['RShoulderRoll'], dFrames['RElbowRoll'], color, linewidth)
    o_RElbow_RWrist = plot3DLine(ax, dFrames['RElbowRoll'], dFrames['RWristYaw'], color, linewidth)
    o_Rwrist_RHand = plot3DLine(ax, dFrames['RWristYaw'], dFrames['RArm'], color, linewidth)
    o_LShoulder_LElbow = plot3DLine(ax, dFrames['LShoulderRoll'], dFrames['LElbowRoll'], color, linewidth)
    o_LElbow_LWrist = plot3DLine(ax, dFrames['LElbowRoll'], dFrames['LWristYaw'], color, linewidth)
    o_LWrist_LHand = plot3DLine(ax, dFrames['LWristYaw'], dFrames['LArm'], color, linewidth)
    o_RShoulder_LShoulder = plot3DLine(ax, dFrames['RShoulderRoll'], dFrames['LShoulderRoll'], color, linewidth)
    
    #plotting low extremity
    o_Shoulder_HipPitch = plot3DLine(ax, shoulder_center, dFrames['HipPitch'], color, linewidth)
    o_HipPitch_KneePitch = plot3DLine(ax, dFrames['HipPitch'], dFrames['KneePitch'], color, linewidth)
    o_KneePitch_Leg = plot3DLine(ax, dFrames['KneePitch'], dFrames['Leg'], color, linewidth)
    
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

def plotEgoSphere(ax, center, cmap, res, radius):
    #theta, phi = np.linspace(0, 2 * np.pi, 12), np.linspace(0, np.pi, 12)
    theta, phi = np.linspace(0, np.pi, res), np.linspace(0, np.pi, res)
    THETA, PHI = np.meshgrid(theta, phi)
    THETA -= np.pi/2.0    
    R = radius
    X = R * np.sin(PHI) * np.cos(THETA) + center[0]
    Y = R * np.sin(PHI) * np.sin(THETA) + center[1]
    Z = R * np.cos(PHI) + center[2]
    C = Y * 0.0    
    scamap = plt.cm.ScalarMappable(cmap=cmap)
    fcolors = scamap.to_rgba(C)
    obj = ax.plot_surface(X, Y, Z, facecolors=fcolors, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.7)    
    return {'obj': obj, 'scamap' : scamap, 'X': X, 'Y': Y, 'Z': Z , 'C' : C}
    #return ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, color='red', antialiased=False, alpha=0.3), scamap

def getij(p, res):
        radToIndex = (res*1.0)/math.pi
        p1 = p[1]
        angle = math.atan2(p[1],p[2])
        print("p", p, "angle", (angle/math.pi)*180.0 )
        #i = (math.atan2(p1,p[2]))* radToIndex - 1.0
        i = (angle)* radToIndex - 1.0
        #j = (math.atan2(p1,p[0]) + np.pi/2.0)* radToIndex - 1.0
        j = (math.atan2(p1,p[0]))* radToIndex - 1.0
        return i,j

def render(motion_service, ax, objRobot, objEgoSph, rightArmEgo, leftArmEgo, network, radius):
    
    dFrames = getRobotFramesPos(motion_service)

    #plotting head 
    shoulder_center = (dFrames['RShoulderRoll'] + dFrames['LShoulderRoll']) / 2.0
    set_data_3DLine(objRobot['Shoulder_Neck'], shoulder_center, dFrames['HeadYaw'])
    set_data_3DLine(objRobot['Neck_HeadTouch'], dFrames['HeadYaw'], dFrames['HeadTouch'])

    #plotting arms
    set_data_3DLine(objRobot['RShoulder_RElbow'],dFrames['RShoulderRoll'], dFrames['RElbowRoll'])
    set_data_3DLine(objRobot['RElbow_RWrist'], dFrames['RElbowRoll'], dFrames['RWristYaw'])
    set_data_3DLine(objRobot['RWrist_RHand'], dFrames['RWristYaw'], dFrames['RArm'])
    set_data_3DLine(objRobot['LShoulder_LElbow'], dFrames['LShoulderRoll'], dFrames['LElbowRoll'])
    set_data_3DLine(objRobot['LElbow_LWrist'], dFrames['LElbowRoll'], dFrames['LWristYaw'])
    set_data_3DLine(objRobot['LWrist_LHand'], dFrames['LWristYaw'], dFrames['LArm'])
    set_data_3DLine(objRobot['RShoulder_LShoulder'], dFrames['RShoulderRoll'], dFrames['LShoulderRoll'])
    
    #plotting low extremity
    set_data_3DLine(objRobot['Shoulder_HipPitch'], shoulder_center, dFrames['HipPitch'])
    set_data_3DLine(objRobot['HipPitch_KneePitch'], dFrames['HipPitch'], dFrames['KneePitch'])
    set_data_3DLine(objRobot['KneePitch_Leg'], dFrames['KneePitch'], dFrames['Leg'])

    RElbowRoll = dFrames['RElbowRoll']
    RWristYaw = dFrames['RWristYaw']
    LElbowRoll = dFrames['LElbowRoll']
    LWristYaw = dFrames['LWristYaw']
    #torso = dFrames['Torso']
    pR1, pR2 = intersect(shoulder_center, radius, RElbowRoll, RWristYaw-RElbowRoll)
    pL1, pL2 = intersect(shoulder_center, radius, LElbowRoll, LWristYaw-LElbowRoll)
    # iL, jL = getij(pL1-torso, network._res)
    # iR, jR = getij(pR1-torso, network._res)

    # updating network
    ptL = pL1-shoulder_center
    ptL = np.array([-ptL[1], ptL[0], ptL[2]])
    ptR = pR1-shoulder_center
    ptR = np.array([-ptR[1], ptR[0], ptR[2]])
    iL, jL = getij(ptL, network._res)
    iR, jR = getij(ptR, network._res)
            
    #print(i,j)
    #print(pij)
    network.updateObjects([np.array([iR,jR]), np.array([iL,jL])])
    u_pre, u_sel, o = network.step({'o':[1.0, 1.0], 'l' : 0.0, 'r': 0.0, 'a':0.0, 'b':0.0, 'n': 0.0})

    # plotting ego-sphere         
    X = objEgoSph['X']
    Y = objEgoSph['Y']
    Z = objEgoSph['Z']
    C = objEgoSph['C']
    scamap = objEgoSph['scamap']
    objES = objEgoSph['obj']
    k = 0   
    for i in range (network._res):
        for j in range (network._res):
            C[i,j] = u_pre[k]
            k += 1
    
    objES.remove()
    scamap = plt.cm.ScalarMappable(cmap='viridis')
    fcolors = scamap.to_rgba(C)    
    objEgoSph['obj'] = ax.plot_surface(X, Y, Z, facecolors=fcolors, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.7) 
    
    # # plotting arm-ego intersection point 
    set_data_3DPoint(leftArmEgo, pL1)
    set_data_3DPoint(rightArmEgo, pR1)
    
    ax.figure.canvas.draw()
    #print('render')

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

    # print(motion_service.getBodyNames('Body'))
    # print(motion_service.getSensorNames())

    res = 16
    radius = 0.25

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(10, 10))
    fig.tight_layout()

    # #fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax.view_init(elev=0, azim=0)

    dt = 0.05
    objs_ij = [np.array([-1.0,-1.0]), np.array([-1.0,-1.0])]

    refs = []
    for i in range(res):
        for j in range(res):
            refs.append(np.array([i,j]))
    refs = np.vstack(refs)
    params = {\
        'ref': refs,
        'res' : res,
        'sig': [[0.5,0.0],[0.0,0.5]],
        #'h' : -0.025,
        'h_pre' : -0.01,
        'h_sel' : -0.0001,
        'dt' : dt,
        'inh' : 0.001,
        'tau' : 0.2,
        'objects': objs_ij
        }              

    network = Network(params) 
    
    
    #torso = getTransfrom(motion_service,'Torso')[:,3]
    shoulder_center = (getTransfrom(motion_service,'RShoulderRoll')[:,3] + getTransfrom(motion_service,'LShoulderRoll')[:,3] )/ 2.0
    # plotting intersection point
    
    rightArmEgo = plot3DPoint(ax, shoulder_center, 'red')
    leftArmEgo = plot3DPoint(ax, shoulder_center, 'red')

    objRobot = plotRobotBody(motion_service, ax, 'darkcyan', 3.0)

    objEgoSph = plotEgoSphere(ax, shoulder_center, 'viridis', res, radius)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-0.75, 0.75])
    ax.set_ylim([-0.75, 0.75])
    ax.set_zlim([-0.01, 1.5])
    # ax.set_xticks([-0.75, 0.75])
    # ax.set_yticks([-0.75, 0.75])
    # ax.set_zticks([0.0, 1.5])
    ax.set_aspect('equal')
    ax.grid(False)

    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(render, motion_service, ax, objRobot, objEgoSph, rightArmEgo, leftArmEgo, network, radius)
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