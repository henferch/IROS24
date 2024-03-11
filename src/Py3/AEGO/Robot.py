import motion
import numpy as np

class RobotViewer:
    def __init__(self, ut, ax):
        
        self.ut = ut
        self.ax = ax

        self.RShoulderRoll = np.zeros((3,), dtype=np.float32) 
        self.RElbowRoll = np.zeros((3,), dtype=np.float32) 
        self.RWristYaw = np.zeros((3,), dtype=np.float32) 
        self.RArm = np.zeros((3,), dtype=np.float32)
        self.LShoulderRoll = np.zeros((3,), dtype=np.float32) 
        self.LElbowRoll = np.zeros((3,), dtype=np.float32) 
        self.LWristYaw = np.zeros((3,), dtype=np.float32) 
        self.LArm = np.zeros((3,), dtype=np.float32)
        self.HeadYaw = np.zeros((3,), dtype=np.float32)
        self.HeadTouch = np.zeros((3,), dtype=np.float32)
        self.HipPitch = np.zeros((3,), dtype=np.float32) 
        self.KneePitch = np.zeros((3,), dtype=np.float32) 
        self.Leg = np.zeros((3,), dtype=np.float32)
        self.Torso = np.zeros((3,), dtype=np.float32)

        #plotting objects
        self.o_Shoulder_Neck = None
        self.o_Neck_HeadTouch = None
        self.o_RShoulder_RElbow = None
        self.o_RElbow_RWrist = None
        self.o_RWrist_RHand = None
        self.o_LShoulder_LElbow = None
        self.o_LElbow_LWrist = None
        self.o_LWrist_LHand = None
        self.o_RShoulder_LShoulder = None
        self.o_Shoulder_HipPitch = None
        self.o_HipPitch_KneePitch = None
        self.o_KneePitch_Leg = None

        print("Robot model created")

    def getBaseLocation(self):
        return self.Torso
    
    def setFrames(self, data):
        self.RShoulderRoll = data[0,:]
        self.RElbowRoll = data[1,:] 
        self.RWristYaw = data[2,:] 
        self.RArm = data[3,:]
        self.LShoulderRoll  = data[4,:]
        self.LElbowRoll = data[5,:]
        self.LWristYaw = data[6,:]
        self.LArm = data[7,:]
        self.HeadYaw = data[8,:]
        self.HeadTouch = data[9,:]
        self.HipPitch = data[10,:]
        self.KneePitch = data[11,:]
        self.Leg = data[12,:]
        self.Torso = data[13,:]
        
    def getFrames(self):
        return [self.RShoulderRoll,\
                self.RElbowRoll, 
                self.RWristYaw,
                self.RArm,
                self.LShoulderRoll,
                self.LElbowRoll,
                self.LWristYaw,
                self.LArm,
                self.HeadYaw,
                self.HeadTouch,
                self.HipPitch,
                self.KneePitch,
                self.Leg,
                self.Torso]
     
    def plot(self, color, linewidth):

        #plotting head 
        shoulder_center = (self.RShoulderRoll + self.LShoulderRoll) / 2.0
        self.o_Shoulder_Neck = self.ut.plot3DLine(self.ax, shoulder_center, self.HeadYaw, color, linewidth)
        self.o_Neck_HeadTouch = self.ut.plot3DLine(self.ax, self.HeadYaw, self.HeadTouch, color, linewidth)

        #plotting arms
        self.o_RShoulder_RElbow = self.ut.plot3DLine(self.ax, self.RShoulderRoll, self.RElbowRoll, color, linewidth)
        self.o_RElbow_RWrist = self.ut.plot3DLine(self.ax, self.RElbowRoll, self.RWristYaw, color, linewidth)
        self.o_RWrist_RHand = self.ut.plot3DLine(self.ax, self.RWristYaw, self.RArm, color, linewidth)
        self.o_LShoulder_LElbow = self.ut.plot3DLine(self.ax, self.LShoulderRoll, self.LElbowRoll, color, linewidth)
        self.o_LElbow_LWrist = self.ut.plot3DLine(self.ax, self.LElbowRoll, self.LWristYaw, color, linewidth)
        self.o_LWrist_LHand = self.ut.plot3DLine(self.ax, self.LWristYaw, self.LArm, color, linewidth)
        self.o_RShoulder_LShoulder = self.ut.plot3DLine(self.ax, self.RShoulderRoll, self.LShoulderRoll, color, linewidth)
        
        #plotting low extremity
        self.o_Shoulder_HipPitch = self.ut.plot3DLine(self.ax, shoulder_center, self.HipPitch, color, linewidth)
        self.o_HipPitch_KneePitch = self.ut.plot3DLine(self.ax, self.HipPitch, self.KneePitch, color, linewidth)
        self.o_KneePitch_Leg = self.ut.plot3DLine(self.ax, self.KneePitch, self.Leg, color, linewidth)
    
    def getIntersectionPoints(self):
        return self.Torso, self.RElbowRoll, self.RWristYaw, self.LElbowRoll, self.LWristYaw

    def render(self):
    
        #plotting head 
        shoulder_center = (self.RShoulderRoll + self.LShoulderRoll) / 2.0
        self.ut.setPlotData3DLine(self.o_Shoulder_Neck, shoulder_center, self.HeadYaw)
        self.ut.setPlotData3DLine(self.o_Neck_HeadTouch, self.HeadYaw, self.HeadTouch)

        #plotting arms
        self.ut.setPlotData3DLine(self.o_RShoulder_RElbow,self.RShoulderRoll, self.RElbowRoll)
        self.ut.setPlotData3DLine(self.o_RElbow_RWrist, self.RElbowRoll, self.RWristYaw)
        self.ut.setPlotData3DLine(self.o_RWrist_RHand, self.RWristYaw, self.RArm)
        self.ut.setPlotData3DLine(self.o_LShoulder_LElbow, self.LShoulderRoll, self.LElbowRoll)
        self.ut.setPlotData3DLine(self.o_LElbow_LWrist, self.LElbowRoll, self.LWristYaw)
        self.ut.setPlotData3DLine(self.o_LWrist_LHand, self.LWristYaw, self.LArm)
        self.ut.setPlotData3DLine(self.o_RShoulder_LShoulder, self.RShoulderRoll, self.LShoulderRoll)
        
        #plotting low extremity
        self.ut.setPlotData3DLine(self.o_Shoulder_HipPitch, shoulder_center, self.HipPitch)
        self.ut.setPlotData3DLine(self.o_HipPitch_KneePitch, self.HipPitch, self.KneePitch)
        self.ut.setPlotData3DLine(self.o_KneePitch_Leg, self.KneePitch, self.Leg)

