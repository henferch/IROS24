#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Use transformInterpolations Method on Arm"""

import sys
import motion
import math
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from Network import Network
from PlotTools import *
from GeodesicDome import GeodesicDome
from RobotPlot import RobotPlot
from Utils import Utils
import copy 

def render(ax, robots, egoSphs, networks, ut):    
    for robot, network, egosph in zip(robots, networks, egoSphs):
        robot.render(False)
        oWeight = np.ones((network._nObjects,), dtype=np.float32)
        oWeight[3] = 2.1
        u_pre, u_sel, o = network.step({'o': oWeight, 'l' : 0.0, 'r': 0.0, 'a':0.0, 'b':0.0, 'n': 0.0})
        egosph.render(u_pre)
        
    ax.figure.canvas.draw()

def plotRectangle(ax, dim, center, color, alpha):

    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta)*dim[0] + center[0]
    y = np.sin(Phi)*np.sin(Theta)*dim[1] + center[1]
    z = np.cos(Theta)/np.sqrt(2)*dim[2] + center[2]
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def transform(rotz, trans, post):
        transform = np.eye(4, dtype=np.float32)
        alpha = (rotz*180.0)/math.pi
        transform[0:3, 0:3] = np.array([[math.cos(alpha), -math.sin(alpha), 0.0], [math.sin(alpha), math.cos(alpha), 0.0],[0.0,0.0,1.0]])
        transform[0:3, 3] = np.array(trans)
        for k,p in post.items():        
            post[k] = np.matmul(transform, np.array([p[0], p[1], p[2], 1.0]))[0:3]
        return post

def main():
    """
    Use case of transformInterpolations API.
    """
    # Get the services ALMotion & ALRobotPosture.

    ut = Utils.getInstance()
    
    #fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(10, 10))
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(12, 8))
    fig.tight_layout()
    ax.view_init(elev=45, azim=45)

    res = 16
    radius = 0.25

    robot1 = RobotPlot(None, ut)
    robot2 = RobotPlot(None, ut)
    robot3 = RobotPlot(None, ut)

    # standing-up
    post1 = {'LShoulderRoll': np.array([-0.04594395,  0.14973637,  0.90913987]),\
             'RShoulderRoll': np.array([-0.04594417, -0.14974362,  0.9091351 ]), 
             'LWristYaw': np.array([-0.08416211,  0.19674906,  0.58608443]), 
             'KneePitch': np.array([-6.19999738e-03, -7.73070497e-12,  3.33999991e-01]), 
             'Leg': np.array([ 3.72529030e-09, -1.28580636e-11,  2.98023224e-08]), 
             'RArm': np.array([-0.07074575, -0.17324671,  0.51307958]), 
             'RWristYaw': np.array([-0.07693396, -0.19625089,  0.58505774]), 
             'HeadTouch': np.array([-6.63890392e-02, -9.85716004e-04,  1.20134616e+00]), 
             'HeadYaw': np.array([-2.30781846e-02, -4.95453151e-06,  9.91237640e-01]), 
             'LArm': np.array([-0.07928477,  0.17397371,  0.513933  ]), 
             'RElbowRoll': np.array([-0.08397251, -0.181114  ,  0.73412597]), 
             'Torso': np.array([ 6.93188189e-03, -2.22946028e-06,  8.19745958e-01]), 
             'LElbowRoll': np.array([-0.0885928 ,  0.18112931,  0.73520309]), 
             'HipPitch': np.array([-3.24670086e-03, -3.63797881e-12,  6.01983726e-01])}

    # pointing-at (right arm)   
    post2 = {'LShoulderRoll': np.array([-0.04594395,  0.14973637,  0.90913987]),\
             'RShoulderRoll': np.array([-0.04594417, -0.14974362,  0.9091351 ]), 
             'LWristYaw': np.array([0.24111512, 0.29563949, 0.97505939]), 
             'KneePitch': np.array([-6.19999738e-03, -7.73070497e-12,  3.33999991e-01]), 
             'Leg': np.array([ 3.72529030e-09, -1.28580636e-11,  2.98023224e-08]), 
             'RArm': np.array([-0.07074575, -0.17324671,  0.51307958]), 
             'RWristYaw': np.array([-0.07693396, -0.19625089,  0.58505774]), 
             'HeadTouch': np.array([0.00795856, 0.00911435, 1.20331287]), 
             'HeadYaw': np.array([-2.30781846e-02, -4.95453151e-06,  9.91237640e-01]), 
             'LArm': np.array([0.31143695, 0.29562247, 1.00339973]), 
             'RElbowRoll': np.array([-0.08397251, -0.181114  ,  0.73412597]), 
             'Torso': np.array([ 6.93188189e-03, -2.22946028e-06,  8.19745958e-01]), 
             'LElbowRoll': np.array([0.11360706, 0.23572412, 0.92356348]), 
             'HipPitch': np.array([-3.24670086e-03, -3.63797881e-12,  6.01983726e-01])}

    # waving (left arm)
    post3 = {'LShoulderRoll': np.array([-0.04594395,  0.14973637,  0.90913987]),\
             'RShoulderRoll': np.array([-0.04594417, -0.14974362,  0.9091351 ]), 
             'LWristYaw': np.array([-0.01066339,  0.23655394,  0.63008094]), 
             'KneePitch': np.array([-6.19999738e-03, -7.73070497e-12,  3.33999991e-01]), 
             'Leg': np.array([ 3.72529030e-09, -1.28580636e-11,  2.98023224e-08]), 
             'RArm': np.array([ 0.14263038, -0.29777843,  1.10070014]), 
             'RWristYaw': np.array([ 0.1185353 , -0.30204576,  1.02893972]), 
             'HeadTouch': np.array([ 9.82208550e-03, -1.05039042e-03,  1.20322549e+00]), 
             'HeadYaw': np.array([-2.30781846e-02, -4.95453151e-06,  9.91237640e-01]), 
             'LArm': np.array([0.01806946, 0.21364284, 0.56376475]), 
             'RElbowRoll': np.array([ 0.09964857, -0.25644034,  0.88729429]), 
             'Torso': np.array([ 6.93188189e-03, -2.22946028e-06,  8.19745958e-01]), 
             'LElbowRoll': np.array([-0.09613313,  0.22837366,  0.75307685]), 
             'HipPitch': np.array([-3.24670086e-03, -3.63797881e-12,  6.01983726e-01])}

    objects = []
    dim = [0.2,0.2,0.2]
    center = [1.8,1.0,0.0]
    objects.append(np.array(center))
    plotRectangle(ax, dim, center, 'green', 0.1)
    dim = [0.2,0.4,0.2]
    center = [1.8,2.4,0.0]
    objects.append(np.array(center))
    plotRectangle(ax, dim, center, 'blue', 0.1)
    dim = [0.2,0.2,0.2]
    center = [1.8,2.5,0.2]
    objects.append(np.array(center))
    plotRectangle(ax, dim, center, 'orange', 0.1)
    center = [-1.0,0.4,0.0]
    dim = [0.2,0.2,0.8]
    #objects.append(np.array(center))
    plotRectangle(ax, dim, center, 'black', 0.1)
    center = [-1.0,0.4,0.4]
    dim = [0.2,0.2,0.2]
    objects.append(np.array(center))
    plotRectangle(ax, dim, center, 'red', 0.1)

    #objects = [objects[3]]

    post1 = transform(75.0, [1.5,0.0,0.0], post1)
    post2 = transform(-90.0, [1.0,1.0,0.0], post2)
    post3 = transform(90.0, [1.0,2.0,0.0], post3)

    robot1.setRobotFramesPos(post1)
    robot2.setRobotFramesPos(post2)
    robot3.setRobotFramesPos(post3)

    #center1 = robot1.getBaseLocation()
    center1 = post1['Torso']
    robot1.plot(ax,'darkcyan', 2.0, False)
    #center2 = robot2.getBaseLocation()
    center2 = post2['Torso']
    robot2.plot(ax,'darkcyan', 2.0, False)
    #center3 = robot3.getBaseLocation()
    center3 = post3['Torso']
    robot3.plot(ax,'darkcyan', 2.0, False)
    
    # Icosahedron
    tesselation = 3        
    sigma = 0.001
    #objects = [np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0])]
    
    params = {'tesselation': tesselation, 'scale' : radius, 'center': center1}            
    egoSphere1 = GeodesicDome(params)
    egoSphere2 = copy.copy(egoSphere1)
    egoSphere2.center = center2
    egoSphere3 = copy.copy(egoSphere1)
    egoSphere3.center = center3

    # plotting intersection points
    pointsEgos = []
    for e,c in zip([egoSphere1, egoSphere2, egoSphere3],[center1, center2, center3]):
        pEgo = []
        for o in objects:
            p1, p2 = e.intersect(o, o-c)
            #ut.plot3DPoint(ax, p1, 'red')
            pEgo.append(p1-c)
        pointsEgos.append(pEgo)

    #plotting projection lines
    for e, c in zip(pointsEgos, [center1, center2, center3]):
        for i in range(len(objects)):
            ut.plot3DLine(ax, e[i]+c, objects[i], 'gray', 0.3, 'dashdot')
        
    refs = egoSphere1.getV()
    dt = 0.05        
    params = {\
        'ut': ut,
        'ref': refs,
        'res' : res,
        'sig': np.eye(3)*sigma,
        'h_pre' : -0.01,
        'h_sel' : -0.0001,
        'dt' : dt,
        'inh' : 0.0001,
        'tau' : 0.2,
        'objects': pointsEgos[0]
        }              

    network1 = Network(params) 
        
    params['ref'] = egoSphere2.getV()
    params['objects'] = pointsEgos[1]
    network2 = Network(params) 

    params['ref'] = egoSphere3.getV()
    params['objects'] = pointsEgos[2]
    network3 = Network(params) 

    egoSphere1.plot(ax, network1.getU_pre(), 0.5)
    egoSphere2.plot(ax, network2.getU_pre(), 0.5)
    egoSphere3.plot(ax, network3.getU_pre(), 0.5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-0.75, 2.5])
    ax.set_ylim([-0.75, 2.5])
    ax.set_zlim([-0.01, 1.5])
    ax.grid(False)
    ax.axis('off')
    # ax.set_xticks([-0.75, 0.75])
    # ax.set_yticks([-0.75, 0.75])
    # ax.set_zticks([0.0, 1.5])
    #ax.set_aspect('equal')
    #ax.grid(False)

    robots = [robot1, robot2, robot3]
    egoSpheres = [egoSphere1, egoSphere2, egoSphere3]
    networks = [network1, network2, network3]
    # robots = [robot1]
    # egoSpheres = [egoSphere1]
    # networks = [network1]


    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(render, ax, robots, egoSpheres, networks, ut)
    timer.start()

    plt.show()
    
    print("plot end")
    timer.stop()
   
    #time.sleep(2.0)

    # Go to rest position
    #motion_service.rest()


if __name__ == "__main__":
    main()