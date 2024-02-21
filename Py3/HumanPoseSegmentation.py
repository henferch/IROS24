import cv2
import mediapipe as mp
import numpy as np 
import time

class HumanPoseSegmentation:
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

    def step(self, image):
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
            image = cv2.flip(image, 1)
    
        t2 = self.getTimeInMS()
        print("Loop time in ms : {}".format(t2 - t1))
    
        return image, self.landmarks