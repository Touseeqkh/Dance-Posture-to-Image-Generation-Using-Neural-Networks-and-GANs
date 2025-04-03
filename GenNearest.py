import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



import numpy as np
import cv2

class GenNeirest:
    """ Class to generate an image from videoSke using a skeleton posture and nearest neighbor method """
    def _init_(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
        """ Generate an image from the skeleton """
        min_distance = float('inf')
        closest_image = None

        for i in range(self.videoSkeletonTarget.skeCount()):
            target_ske = self.videoSkeletonTarget.ske[i]
            distance = ske.distance(target_ske)
            if distance < min_distance:
                min_distance = distance
                closest_image = self.videoSkeletonTarget.readImage(i)
        
        # Ensure that the closest_image is resized to 64x64
        if closest_image is not None:
            resized_image = cv2.resize(closest_image, (64, 64))
            return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Fallback if no image found
        empty = np.ones((64, 64, 3), dtype=np.uint8)
        return empty