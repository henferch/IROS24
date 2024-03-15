
import posix_ipc as pos
import sys
import numpy as np
import mmap
from AEGO.NetworkGeodesic import NetworkGeodesic
from AEGO.GeodesicDome import GeodesicDome
from AEGO.Robot import RobotViewer
from AEGO.Human import HumanViewer
from AEGO.Utils import Utils
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import json

class GUI():
    def __init__(self, params={}):
        
        self.user = params['expID']
        self.ut = Utils.getInstance()

        self.fig, self.axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(14, 7))
        self.fig.tight_layout()
        
        # ExpData shared memory
        nValues = 30 
        singlePrecisionInBytes = 4
        bJointSize = nValues * 3 * singlePrecisionInBytes
        mem = pos.SharedMemory('/{}_expData'.format(self.user), pos.O_CREAT,size=bJointSize)
        self.memExpData = mmap.mmap(mem.fd, bJointSize)
        self.sizeMemExpData = mem.size
        print("memPosture size in bytes: {}".format(self.sizeMemExpData))

        self.memExpData.seek(0)
        bufPos = self.memExpData.read(self.sizeMemExpData)
        data = np.frombuffer(bufPos, dtype=np.float32).reshape((30,3))

        # agent's posture initialisation 

        self.robot = RobotViewer(self.ut, self.axs[0])
        self.human = HumanViewer(self.ut, self.axs[1])

        dataRobot = np.array(data[0:14,:])
        self.robot.setFrames(dataRobot)
        self.robotBaseFramePos = self.robot.getBaseLocation()
        
        dataHuman = np.array(data[14:30,:])
        dataHuman[:,0] *= -1.0
        self.human.setBaseFrame(np.array([0.0, 0.0, params['humanBaseFramePos'][2]]))
        self.human.setFrames(dataHuman)
        self.humanBaseFramePos = self.human.getBaseLocation()

        self.robot.plot('darkcyan', 3.0)
        self.human.plot('darkcyan', 3.0)

        # set neural networks 

        pNn = params['NeuralNetwork']
        self.egoSphereRobot = None
        self.egoSphereHuman = None
        self.networkRobot = None
        self.networkHuman = None
        for n, pNn, bf, ax in zip( ['r', 'h'], 
                                [pNn['robot'], pNn['human']],\
                                [self.robotBaseFramePos, self.humanBaseFramePos],
                                self.axs):
            pNnGd = pNn['GeoDome']
            pNnGd['center'] = bf 
            ego = GeodesicDome(pNnGd)
            pNn['objects'] = [bf, bf]
            pNn['ut'] = self.ut
            pNn['sigma'] = pNn['sigma']*np.eye(3, dtype=np.float32)
            pNn['dt'] = params['dt']
            pNn['ref'] = ego.getV()
            net = NetworkGeodesic(pNn)
            if n == 'r':
                self.egoSphereRobot = ego
                self.networkRobot = net
            else:
                self.egoSphereHuman = ego
                self.networkHuman = net
            ego.plot(ax, net.getU_pre())

        # pNnH = pNn['human']
        # pNnHGd = pNnH['GeoDome']
        # pNnHGd['center'] = self.humanBaseFramePos
        # self.egoSphereHuman = GeodesicDome(pNnHGd)
        # pNnH['objects'] = [np.array(self.humanBaseFramePos), np.array(self.humanBaseFramePos)]
        # pNnH['ut'] = self.ut
        # pNnH['dt'] = params['dt']
        # pNnH['ref'] = self.egoSphereHuman.getV()
        # self.networkHuman = NetworkGeodesic(pNnH)

        # plotting a dummy intersection point        
        self.o_rightArmEgoRobot = self.ut.plot3DPoint(self.axs[0], self.robotBaseFramePos, 'red')
        self.o_leftArmEgoRobot = self.ut.plot3DPoint(self.axs[0], self.robotBaseFramePos, 'red')
        self.o_rightArmEgoHuman = self.ut.plot3DPoint(self.axs[1], self.humanBaseFramePos, 'red')
        self.o_leftArmEgoHuman = self.ut.plot3DPoint(self.axs[1], self.humanBaseFramePos, 'red')
        
        i = 0
        for ax in self.axs :
            ax.view_init(elev=0, azim=0)
            # if i == 0:
            #     ax.view_init(elev=0, azim=0)
            # else:
            #     ax.view_init(elev=0, azim=180)
            # i += 1
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim([-0.75, 0.75])
            ax.set_ylim([-0.75, 0.75])
            ax.set_zlim([-0.01, 1.8])
            ax.set_xticks([-0.75, 0.75])
            ax.set_yticks([-0.75, 0.75])
            ax.set_zticks([0.0, 1.8])
            ax.set_aspect('equal')
            ax.grid(False)
        
        self.timer = self.fig.canvas.new_timer(interval=params['attentionTrackerPeriodInMs'])
        self.timer.add_callback(self.render)
        self.timer.start()

        plt.show()

        print("Viewer end")
        self.timer.stop()
               
    def setData(self, robot, human):
        self.robotData = robot
        self.humanData = human

    def stop(self):
        print("plot end")
        self.timer.stop()
        plt.close()
        
    def render(self):

        self.memExpData.seek(0)
        bufPos = self.memExpData.read(self.sizeMemExpData)
        data = np.frombuffer(bufPos, dtype=np.float32).reshape((30,3))

        dataRobot = np.array(data[0:14,:])
        dataHuman = np.array(data[14:30,:])
        dataHuman[:,0] *= -1.0
        
        self.robot.setFrames(dataRobot)
        self.human.setFrames(dataHuman)

        self.robot.render()
        self.human.render()

        for agent,ego,net in zip([self.robot, self.human], [self.egoSphereRobot, self.egoSphereHuman], [self.networkRobot, self.networkHuman]): 
            
            Torso, RElbow, RWrist, LElbow, LWrist = agent.getIntersectionPoints()
        
            pR1, pR2 = ego.intersect(RElbow, RWrist-RElbow)
            pL1, pL2 = ego.intersect(LElbow, LWrist-LElbow)

            if not (pR1 is None or pL1 is None):  
                net.updateObjects([pR1-Torso, pL1-Torso])

            # dRC = np.dot(ego.center, RWrist)
            # dLC = np.dot(ego.center, RWrist)

            hw = None
            if pR1[2] < pL1[2]:
                hw = [10.0, 30.0]
            else:
                hw = [30.0, 10.0]
                
            u_pre, u_sel, o = net.step({'o':hw, 'l' : 0.0, 'r': 0.0, 'a':0.0, 'b':0.0, 'n': 0.0})
            #u_pre, u_sel, o = net.step({'o':[20.0, 20.0], 'l' : 0.0, 'r': 0.0, 'a':0.0, 'b':0.0, 'n': 0.0})

            #ego.render(u_pre)
            ego.render(u_sel)

        
        for ax in self.axs :
            ax.figure.canvas.draw()


if __name__ == "__main__":

    # Opening JSON file
    params = None
    try:
        f = open('/home/hfchame/Workspace/VSCode/IROS24/src/parameters.json')
        params = json.load(f)
        f.close()
    except Exception as ex:
        print ("Can't load parameters from file 'parameters.json'. Error: {}".format(ex))
        sys.exit(1)

    viewer = None 
    try:
        viewer = GUI(params)
    except KeyboardInterrupt:
        print("Program stopped by user")
        plt.close()
        sys.exit(1)
