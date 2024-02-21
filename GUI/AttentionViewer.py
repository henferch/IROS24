from mpl_toolkits import mplot3d
import numpy as np
import json
import matplotlib.pyplot as plt
import mmap
import posix_ipc as pos

class AttentionViewer():
    def __init__(self, params={}):
        
        self.user = "ManipHFC2024"
        fig = plt.figure()
        self.ax = plt.axes(projection='3d')

        self.robotData = []
        self.humanData = []

        self.robotColor = 'blue'
        self.humanColor = 'blue'

          # ExpData shared memory
        nValues = 30 
        singlePrecisionInBytes = 4
        bJointSize = nValues * 3 * singlePrecisionInBytes
        mem = pos.SharedMemory('/{}_expData'.format(self.user), pos.O_CREAT,size=bJointSize)
        self.memExpData = mmap.mmap(mem.fd, bJointSize)
        self.sizeMemExpData = mem.size
        print("memPosture size in bytes: {}".format(self.sizeMemExpData))

        # robot objects
        self.robotObjects_1 = self.ax.plot3D([], [], [], self.robotColor)[0]
        self.robotObjects_2 = self.ax.plot3D([], [], [], self.robotColor)[0]
        self.robotObjects_3 = self.ax.plot3D([], [], [], self.robotColor)[0]
        self.robotObjects_4 = self.ax.plot3D([], [], [], self.robotColor)[0]
        self.robotObjects_5 = self.ax.plot3D([], [], [], self.robotColor)[0]
        self.robotObjects_6 = self.ax.plot3D([], [], [], self.robotColor)[0]
        self.robotObjects_7 = self.ax.plot3D([], [], [], self.robotColor)[0]
        self.robotObjects_8 = self.ax.plot3D([], [], [], self.robotColor)[0]
        self.robotObjects_9 = self.ax.plot3D([], [], [], self.robotColor)[0]
        self.robotObjects_10 = self.ax.plot3D([], [], [], self.robotColor)[0]
        self.robotObjects_11 = self.ax.plot3D([], [], [], self.robotColor)[0]
        self.robotObjects_12 = self.ax.plot3D([], [], [], self.robotColor)[0]

        # plotting ego spheres
        theta, phi = np.linspace(0, 2 * np.pi, 12), np.linspace(0, np.pi, 12)
        self.THETA, self.PHI = np.meshgrid(theta, phi)
        self.R = 0.25
        X = self.R * np.sin(self.PHI) * np.cos(self.THETA)
        Y = self.R * np.sin(self.PHI) * np.sin(self.THETA)
        Z = self.R * np.cos(self.PHI)
        self.egoSRobotObj = self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, color='red', antialiased=False, alpha=0.3)        
        self.egoSHumanObj = self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, color='red', antialiased=False, alpha=0.3)

        # human objects
        self.humanObjects_1 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_2 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_3 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_4 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_5 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_6 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_7 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_8 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_9 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_10 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_11 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_12 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_13 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_14 = self.ax.plot3D([], [], [], self.humanColor)[0]
        self.humanObjects_15 = self.ax.plot3D([], [], [], self.humanColor)[0]

        self.ax.set_xlim([-0.75, 1.75])
        self.ax.set_ylim([-0.75, 1.75])
        self.ax.set_zlim([-0.01, 1.5])
        self.ax.set_xticks([-0.75, 0.75])
        self.ax.set_yticks([-0.75, 0.75])
        self.ax.set_zticks([0.0, 1.5])
        #self.ax.set_aspect('equal')
        self.ax.grid(False)
        self.timer = fig.canvas.new_timer(interval=params['AttentionViewerPeriodInMs'])
        self.timer.add_callback(self.render)
        self.timer.start()

        plt.show()
               

    def setData(self, robot, human):
        self.robotData = robot
        self.humanData = human

    def update_Line3D(self, obj, P1, P2):
        obj.set_data([P1[0],P2[0]], [P1[1],P2[1]])
        obj.set_3d_properties([P1[2],P2[2]])

    def stop(self):
        print("plot end")
        self.timer.stop()
        plt.close()
        
    def render(self):

        self.memExpData.seek(0)
        bufPos = self.memExpData.read(self.sizeMemExpData)
        data = np.frombuffer(bufPos, dtype=np.float32).reshape((30,3))

        RShoulderRoll = data[0,:]
        RElbowRoll = data[1,:]
        RWristYaw = data[2,:]
        RArm = data[3,:]
        LShoulderRoll = data[4,:]
        LElbowRoll = data[5,:]
        LWristYaw = data[6,:]
        LArm = data[7,:]
        HeadYaw = data[8,:]
        Head_Touch_Front = data[9,:]
        HipPitch = data[10,:]
        KneePitch = data[11,:]
        Leg = data[12,:]
        torso = data[13,:]

        shoulder_center = (RShoulderRoll + LShoulderRoll) / 2.0 
        self.update_Line3D(self.robotObjects_1, shoulder_center, HeadYaw)
        self.update_Line3D(self.robotObjects_2, HeadYaw, Head_Touch_Front)
        self.update_Line3D(self.robotObjects_3, RShoulderRoll, RElbowRoll)
        self.update_Line3D(self.robotObjects_4, RElbowRoll, RWristYaw)
        self.update_Line3D(self.robotObjects_5, RWristYaw, RArm)
        self.update_Line3D(self.robotObjects_6, LShoulderRoll, LElbowRoll)
        self.update_Line3D(self.robotObjects_7, LElbowRoll, LWristYaw)
        self.update_Line3D(self.robotObjects_8, LWristYaw, LArm)
        self.update_Line3D(self.robotObjects_9, RShoulderRoll, LShoulderRoll)
        self.update_Line3D(self.robotObjects_10, shoulder_center, HipPitch)
        self.update_Line3D(self.robotObjects_11, HipPitch, KneePitch)
        self.update_Line3D(self.robotObjects_12, KneePitch, Leg)

        X = self.R * np.sin(self.PHI) * np.cos(self.THETA) + torso[0] 
        Y = self.R * np.sin(self.PHI) * np.sin(self.THETA) + torso[1]
        Z = self.R * np.cos(self.PHI) + torso[2]

        self.egoSRobotObj.remove()
        self.egoSRobotObj = self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, color='red', antialiased=False, alpha=0.3)        
            
        right_eye_inner = data[14,:]
        right_shoulder = data[15,:]
        right_elbow = data[16,:]
        right_wrist = data[17,:]
        right_index = data[18,:]
        right_hip = data[19,:]
        right_knee = data[20,:]
        right_ancle = data[21,:]
        
        left_eye_inner = data[22,:] 
        left_shoulder = data[23,:]
        left_elbow = data[24,:]
        left_wrist = data[25,:]
        left_index = data[26,:]
        left_hip = data[27,:]
        left_knee = data[28,:]
        left_ancle = data[29,:]

        torso = (left_hip + right_hip + left_shoulder + right_shoulder) / 4.0

        forehead = (right_eye_inner + left_eye_inner) / 2.0
        shoulder_center = (right_shoulder + left_shoulder) / 2.0
        self.update_Line3D(self.humanObjects_1, shoulder_center, forehead)
        self.update_Line3D(self.humanObjects_2, right_shoulder, left_shoulder)
        self.update_Line3D(self.humanObjects_3, right_shoulder, right_elbow)
        self.update_Line3D(self.humanObjects_4, right_elbow, right_wrist)
        self.update_Line3D(self.humanObjects_5, right_wrist, right_index)
        self.update_Line3D(self.humanObjects_6, right_shoulder, right_hip)
        self.update_Line3D(self.humanObjects_7, right_hip, right_knee)
        self.update_Line3D(self.humanObjects_8, right_knee, right_ancle)
        self.update_Line3D(self.humanObjects_9, left_shoulder, left_elbow)
        self.update_Line3D(self.humanObjects_10, left_elbow, left_wrist)
        self.update_Line3D(self.humanObjects_11, left_wrist, left_index)
        self.update_Line3D(self.humanObjects_12, left_shoulder, left_hip)
        self.update_Line3D(self.humanObjects_13, left_hip, left_knee)
        self.update_Line3D(self.humanObjects_14, left_knee, left_ancle)

        X = self.R * np.sin(self.PHI) * np.cos(self.THETA) + torso[0] 
        Y = self.R * np.sin(self.PHI) * np.sin(self.THETA) + torso[1]
        Z = self.R * np.cos(self.PHI) + torso[2]
        self.egoSHumanObj.remove()
        self.egoSHumanObj = self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, color='red', antialiased=False, alpha=0.3)        

        self.ax.figure.canvas.draw()


if __name__ == "__main__":    
    # Opening JSON file
    f = open('parameters.json')
    parameters = json.load(f)
    f.close()
    viewer = None 
    try:
        viewer = AttentionViewer(parameters)
    except KeyboardInterrupt:
        print("Program stopped by user")
        viewer.stop()
