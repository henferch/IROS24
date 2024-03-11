import cv2
import mediapipe as mp
import numpy as np 
import time

class HumanViewer:
    def __init__(self, ut, ax):

        self.ut = ut
        self.ax = ax
        self.hip_height = np.zeros((3,), dtype=np.float32) 
        
        # detected frames
        self.right_eye_inner = np.zeros((3,), dtype=np.float32)
        self.right_shoulder = np.zeros((3,), dtype=np.float32)
        self.right_elbow = np.zeros((3,), dtype=np.float32)
        self.right_wrist = np.zeros((3,), dtype=np.float32)
        self.right_index = np.zeros((3,), dtype=np.float32)
        self.right_hip = np.zeros((3,), dtype=np.float32)
        self.right_knee = np.zeros((3,), dtype=np.float32)
        self.right_ancle = np.zeros((3,), dtype=np.float32)
        self.left_eye_inner = np.zeros((3,), dtype=np.float32)
        self.left_shoulder = np.zeros((3,), dtype=np.float32)
        self.left_elbow = np.zeros((3,), dtype=np.float32)
        self.left_wrist = np.zeros((3,), dtype=np.float32)
        self.left_index = np.zeros((3,), dtype=np.float32)
        self.left_hip = np.zeros((3,), dtype=np.float32)
        self.left_knee = np.zeros((3,), dtype=np.float32)
        self.left_ancle = np.zeros((3,), dtype=np.float32)
        # virtual frames
        self.torso = np.zeros((3,), dtype=np.float32)
        self.forehead = np.zeros((3,), dtype=np.float32)
        self.hip = np.zeros((3,), dtype=np.float32)
        self.knee = np.zeros((3,), dtype=np.float32)
        self.leg = np.zeros((3,), dtype=np.float32)
        self.base = np.zeros((3,), dtype=np.float32)

        #plotting objects
        self.o_torso_head = None
        self.o_lr_shoulder = None
        self.o_r_shoulder_elbow = None
        self.o_r_elbow_wrist = None
        self.o_r_wrist_index = None
        self.o_l_shoulder_elbow = None
        self.o_l_elbow_wrist = None
        self.o_l_wrist_index = None
        self.o_torso_hip = None
        self.o_hip_knee = None
        self.o_knee_leg = None
    
    def setBaseFrame(self, v):
        self.base = v

    def getBaseLocation(self):
        return self.torso
    
    def getIntersectionPoints(self):
        return self.torso, self.right_elbow, self.right_wrist, self.left_elbow, self.left_wrist
    
    def getFrames(self):
        return [self.right_eye_inner,\
                self.right_shoulder,
                self.right_elbow,
                self.right_wrist,
                self.right_index,
                self.right_hip,
                self.right_knee,
                self.right_ancle,
                self.left_eye_inner,
                self.left_shoulder,
                self.left_elbow,
                self.left_wrist, 
                self.left_index, 
                self.left_hip,
                self.left_knee,
                self.left_ancle,
                self.torso]

    def setFrames(self, data_):
        data = np.array(data_) + self.base
        self.right_eye_inner = data[0,:]
        self.right_shoulder = data[1,:]
        self.right_elbow = data[2,:]
        self.right_wrist = data[3,:]
        self.right_index = data[4,:]
        self.right_hip = data[5,:]
        self.right_knee = data[6,:]
        self.right_ancle = data[7,:]
        self.left_eye_inner = data[8,:] 
        self.left_shoulder = data[9,:]
        self.left_elbow = data[10,:]
        self.left_wrist = data[11,:]
        self.left_index = data[12,:]
        self.left_hip = data[13,:]
        self.left_knee = data[14,:]
        self.left_ancle = data[15,:]
        #virtual frames
        #self.torso = (self.right_shoulder + self.left_shoulder) / 2.0
        self.torso = 0.35*self.right_shoulder + 0.35*self.left_shoulder + 0.15*self.left_hip + 0.15*self.right_hip 
        self.forehead = (self.right_eye_inner + self.left_eye_inner) / 2.0
        self.hip = (self.right_hip + self.left_hip) / 2.0
        self.knee = (self.right_knee + self.left_knee) / 2.0
        self.leg = (self.right_ancle + self.left_ancle) / 2.0

    def plot(self, color, linewidth):
        self.o_torso_head = self.ut.plot3DLine(self.ax, self.torso, self.forehead, color, linewidth)
        self.o_lr_shoulder = self.ut.plot3DLine(self.ax, self.right_shoulder, self.left_shoulder, color, linewidth)
        self.o_r_shoulder_elbow = self.ut.plot3DLine(self.ax, self.right_shoulder, self.right_elbow, color, linewidth)
        self.o_r_elbow_wrist = self.ut.plot3DLine(self.ax, self.right_elbow, self.right_wrist, color, linewidth)
        self.o_r_wrist_index = self.ut.plot3DLine(self.ax, self.right_wrist, self.right_index, color, linewidth)
        self.o_l_shoulder_elbow = self.ut.plot3DLine(self.ax, self.left_shoulder, self.left_elbow, color, linewidth)
        self.o_l_elbow_wrist = self.ut.plot3DLine(self.ax, self.left_elbow, self.left_wrist, color, linewidth)
        self.o_l_wrist_index = self.ut.plot3DLine(self.ax, self.left_wrist, self.left_index, color, linewidth)
        self.o_torso_hip = self.ut.plot3DLine(self.ax, self.torso, self.hip, color, linewidth)
        self.o_hip_knee = self.ut.plot3DLine(self.ax, self.hip, self.knee, color, linewidth)
        self.o_knee_leg = self.ut.plot3DLine(self.ax, self.knee, self.leg, color, linewidth)
    
    def render(self):
        self.ut.setPlotData3DLine(self.o_torso_head, self.torso, self.forehead)
        self.ut.setPlotData3DLine(self.o_lr_shoulder, self.right_shoulder, self.left_shoulder)
        self.ut.setPlotData3DLine(self.o_r_shoulder_elbow, self.right_shoulder, self.right_elbow)
        self.ut.setPlotData3DLine(self.o_r_elbow_wrist, self.right_elbow, self.right_wrist)
        self.ut.setPlotData3DLine(self.o_r_wrist_index, self.right_wrist, self.right_index)
        self.ut.setPlotData3DLine(self.o_l_shoulder_elbow, self.left_shoulder, self.left_elbow)
        self.ut.setPlotData3DLine(self.o_l_elbow_wrist, self.left_elbow, self.left_wrist)
        self.ut.setPlotData3DLine(self.o_l_wrist_index, self.left_wrist, self.left_index)
        self.ut.setPlotData3DLine(self.o_torso_hip, self.torso, self.hip)
        self.ut.setPlotData3DLine(self.o_hip_knee, self.hip, self.knee)
        self.ut.setPlotData3DLine(self.o_knee_leg, self.knee, self.leg)
        

class HumanTracker:
    def __init__(self):

        ## Mediapipe Doc API https://google.github.io/mediapipe/solutions/pose.html
    
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        self.landMarkLabels = ['nose',\
                        'left_eye_inner', 
                        'left_eye', 
                        'left_eye_outer',
                        'right_eye_inner', 
                        'right_eye', 
                        'right_eye_outer',
                        'left_ear',
                        'right_ear',
                        'mouth_left',
                        'mouth_right',
                        'left_shoulder',
                        'right_shoulder',
                        'left_elbow',
                        'right_elbow',
                        'left_wrist',
                        'right_wrist',
                        'left_pinky',
                        'right_pinky',
                        'left_index',
                        'right_index',
                        'left_thumb',
                        'right_thumb',
                        'left_hip',
                        'right_hip',
                        'left_knee',
                        'right_knee',
                        'left_ankle',
                        'right_ankle',
                        'left_heel',
                        'right_heel',
                        'left_foot_index',
                        'right_foot_index']
        
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)      
        self.landmarks = np.zeros((len(self.landMarkLabels),4)) 
        
    def getTimeInMS(self):
        return round(time.time()*1000)

    def step(self, image, selfie = False):
        t1 = self.getTimeInMS()
    
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.landmarks = np.zeros((len(self.landMarkLabels),4))
        #if not results.pose_landmarks is None :
        if not results.pose_world_landmarks is None :
            for lId, i in zip(self.landMarkLabels, range(len(self.landMarkLabels))): 
                #l = results.pose_landmarks.landmark[i]
                l = results.pose_world_landmarks.landmark[i]
                self.landmarks[i,:] = np.array([l.x,l.y,l.z,l.visibility])        
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
            # Flip the image horizontally for a selfie-view display.
            if selfie :
                image = cv2.flip(image, 1)
    
        t2 = self.getTimeInMS()
        print("Loop time in ms : {}".format(t2 - t1))
    
        return image, self.landmarks