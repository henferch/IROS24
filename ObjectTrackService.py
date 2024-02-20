# -*- encoding: UTF-8 -*-
import qi
import math
import almath
import numpy as np

class ObjectTrackService:
    def __init__(self, memory_srv, landmark_srv, motion_srv):

        self.memory_srv = memory_srv
        self.landmark_srv = landmark_srv
        self.motion_srv = motion_srv

        # Connect the event callback.
        self.subscriber = self.memory_srv.subscriber("LandmarkDetected")
        self.subscriber.signal.connect(self.on_landmark_detected)        
        self.landmark_srv.subscribe("LandmarkDetector", 500, 0.0 )
        self.got_landmark = False
        self.landmarkTheoreticalSize = 0.093 #in meters
        # Set here the current camera ("CameraTop" or "CameraBottom").
        self.currentCamera = "CameraTop"
        self.landmarks = {}
        
    def on_landmark_detected(self, markData):
        """
        Callback for event LandmarkDetected.
        """

        if markData == []:
            return 

        # Get current camera position in NAO space.
        transform = self.motion_srv.getTransform(self.currentCamera, 2, True)
        transformList = almath.vectorFloat(transform)
        robotToCamera = almath.Transform(transformList)
        
        for lm in markData[1]:
            lId = lm[1][0]
                
            # Retrieve landmark center position in radians.
            wzCamera = lm[0][1]
            wyCamera = lm[0][2]
            # Retrieve landmark angular size in radians.
            angularSize = lm[0][3]

            # Compute distance to landmark.
            distanceFromCameraToLandmark = self.landmarkTheoreticalSize / ( 2 * math.tan( angularSize / 2))

            # Compute the rotation to point towards the landmark.
            cameraToLandmarkRotationTransform = almath.Transform_from3DRotation(0, wyCamera, wzCamera)

            # Compute the translation to reach the landmark.
            cameraToLandmarkTranslationTransform = almath.Transform(distanceFromCameraToLandmark, 0, 0)

            # Combine all transformations to get the landmark position in NAO space.
            robotToLandmark = robotToCamera * cameraToLandmarkRotationTransform *cameraToLandmarkTranslationTransform

            self.landmarks[lId] = np.array([robotToLandmark.r1_c4, robotToLandmark.r2_c4, robotToLandmark.r3_c4])
            
    def getObjects(self):
        return self.landmarks

    def stop(self):
        print ("Désinscription de LandmarkDetector")
        self.landmark_srv.unsubscribe("LandmarkDetector")
        