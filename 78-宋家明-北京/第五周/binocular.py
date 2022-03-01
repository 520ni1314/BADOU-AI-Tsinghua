import numpy as np
import cv2
import argparse
import glob

if __name__=='__main__':
    print('双目系统推导')
    """
    双目系统推导
    
    
    B/Z=pp'/(Z-f)=(B-(Xr-W/2)-(W/2-Xt))/(Z-f)=(B+Xt-Xr)/(Z-f)
    Z(B+Xt-Xr)=B(Z-f)
    Z*Xt-Z*Xr=-B*f
    Z=B*f/(Xr-Xt)
    Xr-Xt=D D为视差，即为已知量。
    Z=B*f/D

    """
