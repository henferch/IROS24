import motion
import numpy as np

class RobotPlot:
    def __init__(self, motion_service, ut):
        
        self.motion_service = motion_service
        self.ut = ut
        self.frame = motion.FRAME_ROBOT #FRAME_TORSO

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

    # unfortunately there is o simple way to get all transforms at once
    def getTransfrom(self, motion_service, name):
        frame  = motion.FRAME_ROBOT #FRAME_TORSO
        useSensorValues  = True
        f = motion_service.getTransform(name, frame, useSensorValues)
        return np.array(f).reshape((4,4))

    def getBaseLocation(self):
        f = self.motion_service.getTransform('Torso', self.frame, True)
        return np.array([f[3],f[7],f[11]])
    
    def setRobotFramesPos(self, dFrames):
        self.RShoulderRoll  = dFrames['RShoulderRoll']
        self.RElbowRoll = dFrames['RElbowRoll'] 
        self.RWristYaw = dFrames['RWristYaw'] 
        self.RArm = dFrames['RArm']
        self.LShoulderRoll  = dFrames['LShoulderRoll']
        self.LElbowRoll = dFrames['LElbowRoll']
        self.LWristYaw = dFrames['LWristYaw']
        self.LArm = dFrames['LArm']
        self.HeadYaw = dFrames['HeadYaw']
        self.HeadTouch = dFrames['HeadTouch']
        self.HipPitch  = dFrames['HipPitch']
        self.KneePitch  = dFrames['KneePitch']
        self.Leg = dFrames['Leg']
        self.Torso = dFrames['Torso']
        

    def getRobotFramesPos(self):
        
        self.updateRobotFramesPos()
        dFrames = {} 
        dFrames['RShoulderRoll'] = self.RShoulderRoll 
        dFrames['RElbowRoll'] = self.RElbowRoll 
        dFrames['RWristYaw'] = self.RWristYaw 
        dFrames['RArm'] = self.RArm
        dFrames['LShoulderRoll'] = self.LShoulderRoll 
        dFrames['LElbowRoll'] = self.LElbowRoll 
        dFrames['LWristYaw'] = self.LWristYaw 
        dFrames['LArm'] = self.LArm
        dFrames['HeadYaw'] = self.HeadYaw
        dFrames['HeadTouch'] = self.HeadTouch
        dFrames['HipPitch'] = self.HipPitch 
        dFrames['KneePitch'] = self.KneePitch 
        dFrames['Leg'] = self.Leg
        dFrames['Torso'] = self.Torso
        return dFrames
     
    def updateRobotFramesPos(self):
        
        fRShoulderRoll = self.motion_service.getTransform('RShoulderRoll', self.frame, True) 
        fRElbowRoll = self.motion_service.getTransform('RElbowRoll', self.frame, True)
        fRWristYaw = self.motion_service.getTransform('RWristYaw', self.frame, True)
        fRArm = self.motion_service.getTransform('RArm', self.frame, True)
        fLShoulderRoll = self.motion_service.getTransform('LShoulderRoll', self.frame, True)
        fLElbowRoll = self.motion_service.getTransform('LElbowRoll', self.frame, True)
        fLWristYaw = self.motion_service.getTransform('LWristYaw', self.frame, True)
        fLArm = self.motion_service.getTransform('LArm', self.frame, True)
        fHeadYaw = self.motion_service.getTransform('HeadYaw', self.frame, True)
        fHeadTouch = self.motion_service.getTransform('Head/Touch/Front', self.frame, True)
        fHipPitch = self.motion_service.getTransform('HipPitch', self.frame, True)
        fKneePitch = self.motion_service.getTransform('KneePitch', self.frame, True)
        fLeg = self.motion_service.getTransform('Leg', self.frame, True)
        fTorso = self.motion_service.getTransform('Torso', self.frame, True)

        self.RShoulderRoll = np.array([fRShoulderRoll[3],fRShoulderRoll[7],fRShoulderRoll[11]]) 
        self.RElbowRoll = np.array([fRElbowRoll[3],fRElbowRoll[7],fRElbowRoll[11]])
        self.RWristYaw = np.array([fRWristYaw[3],fRWristYaw[7],fRWristYaw[11]])
        self.RArm = np.array([fRArm[3],fRArm[7],fRArm[11]])
        self.LShoulderRoll = np.array([fLShoulderRoll[3],fLShoulderRoll[7],fLShoulderRoll[11]])
        self.LElbowRoll = np.array([fLElbowRoll[3],fLElbowRoll[7],fLElbowRoll[11]])
        self.LWristYaw = np.array([fLWristYaw[3],fLWristYaw[7],fLWristYaw[11]])
        self.LArm = np.array([fLArm[3],fLArm[7],fLArm[11]])
        self.HeadYaw = np.array([fHeadYaw[3],fHeadYaw[7],fHeadYaw[11]])
        self.HeadTouch = np.array([fHeadTouch[3],fHeadTouch[7],fHeadTouch[11]])
        self.HipPitch = np.array([fHipPitch[3],fHipPitch[7],fHipPitch[11]])
        self.KneePitch = np.array([fKneePitch[3],fKneePitch[7],fKneePitch[11]])
        self.Leg = np.array([fLeg[3],fLeg[7],fLeg[11]])
        self.Torso = np.array([fTorso[3],fTorso[7],fTorso[11]])

    # def plot3DLine(self, ax, P1, P2, color, linewidth):
    #     return ax.plot3D([P1[0],P2[0]], [P1[1],P2[1]], [P1[2],P2[2]], color=color, linewidth=linewidth)[0]
    
    # def set_data_3DLine(self, obj, P1, P2):
    #     obj.set_data([P1[0],P2[0]], [P1[1],P2[1]])
    #     obj.set_3d_properties([P1[2],P2[2]])

    def plot(self, ax, color, linewidth, useSensor=True):

        if useSensor:
            self.getRobotFramesPos()

        #plotting head 
        shoulder_center = (self.RShoulderRoll + self.LShoulderRoll) / 2.0
        self.o_Shoulder_Neck = self.ut.plot3DLine(ax, shoulder_center, self.HeadYaw, color, linewidth)
        self.o_Neck_HeadTouch = self.ut.plot3DLine(ax, self.HeadYaw, self.HeadTouch, color, linewidth)

        #plotting arms
        self.o_RShoulder_RElbow = self.ut.plot3DLine(ax, self.RShoulderRoll, self.RElbowRoll, color, linewidth)
        self.o_RElbow_RWrist = self.ut.plot3DLine(ax, self.RElbowRoll, self.RWristYaw, color, linewidth)
        self.o_RWrist_RHand = self.ut.plot3DLine(ax, self.RWristYaw, self.RArm, color, linewidth)
        self.o_LShoulder_LElbow = self.ut.plot3DLine(ax, self.LShoulderRoll, self.LElbowRoll, color, linewidth)
        self.o_LElbow_LWrist = self.ut.plot3DLine(ax, self.LElbowRoll, self.LWristYaw, color, linewidth)
        self.o_LWrist_LHand = self.ut.plot3DLine(ax, self.LWristYaw, self.LArm, color, linewidth)
        self.o_RShoulder_LShoulder = self.ut.plot3DLine(ax, self.RShoulderRoll, self.LShoulderRoll, color, linewidth)
        
        #plotting low extremity
        self.o_Shoulder_HipPitch = self.ut.plot3DLine(ax, shoulder_center, self.HipPitch, color, linewidth)
        self.o_HipPitch_KneePitch = self.ut.plot3DLine(ax, self.HipPitch, self.KneePitch, color, linewidth)
        self.o_KneePitch_Leg = self.ut.plot3DLine(ax, self.KneePitch, self.Leg, color, linewidth)
    
    def getIntersectionPoints(self):
        return self.Torso, self.RElbowRoll, self.RWristYaw, self.LElbowRoll, self.LWristYaw

    def render(self, useSensor=True):
    
        if useSensor:
            self.getRobotFramesPos()

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

