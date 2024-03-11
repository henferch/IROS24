import numpy as np

class Utils:
    def __init__(self):
        self.name = "Utils"
        
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
