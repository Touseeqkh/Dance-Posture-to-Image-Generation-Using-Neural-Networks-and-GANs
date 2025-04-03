import numpy as np
import cv2
import os
import pickle
import sys

import tp
from VideoSkeleton import VideoSkeleton
from VideoSkeleton import combineTwoImages
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenNearest import GenNearest  # Fixed spelling of GenNearest
from GenVanillaNN import *
from GenGAN import *


class DanceDemo:
    """Class that runs a demo of the dance.
       The animation/posture from self.source is applied to the character defined by self.target using self.gen
    """
    def __init__(self, filename_src, typeOfGen=2):
        self.target = VideoSkeleton("D:\tp_dance_start (2)\dance_start\data\taichi1.mp4")
        self.source = VideoReader(filename_src)
        if typeOfGen == 1:  # Nearest
            print("Generator: GenNearest")
            self.generator = GenNearest(self.target)
        elif typeOfGen == 2:  # VanillaNN
            print("Generator: GenVanillaNN")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=1)
        elif typeOfGen == 3:  # VanillaNN
            print("Generator: GenVanillaNN")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=2)
        elif typeOfGen == 4:  # GAN
            print("Generator: GenGAN")
            self.generator = GenGAN(self.target, loadFromFile=True)
        else:
            print("DanceDemo: Invalid typeOfGen value!")

    def draw(self):
        ske = Skeleton()
        image_err = np.zeros((256, 256, 3), dtype=np.uint8)
        image_err[:, :] = (0, 0, 255)  # Red color for error image
        for i in range(self.source.getTotalFrames()):
            image_src = self.source.readFrame()
            if i % 5 == 0:
                isSke, image_src, ske = self.target.cropAndSke(image_src, ske)
                if isSke:
                    ske.draw(image_src)
                    image_tgt = self.generator.generate(ske)  # GENERATOR !!!
                    image_tgt = image_tgt * 255
                    image_tgt = cv2.resize(image_tgt, (128, 128))
                else:
                    image_tgt = image_err
                image_combined = combineTwoImages(image_src, image_tgt)
                image_combined = cv2.resize(image_combined, (512, 256))
                cv2.imshow('Image', image_combined)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                if key & 0xFF == ord('n'):
                    self.source.readNFrames(100)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Define constants for generator types
    NEAREST = 1
    VANILLA_NN_SKE = 2
    VANILLA_NN_IMAGE = 3
    GAN = 4
    GEN_TYPE = 1
    
    # Initialize the DanceDemo with the selected generator type
    ddemo = DanceDemo("data/taichi2_full.mp4", GEN_TYPE)
    
    # Uncomment the lines below to use different video sources
    # ddemo = DanceDemo("data/taichi1.mp4")
    # ddemo = DanceDemo("data/karate1.mp4")
    
    # Run the demo
    ddemo.draw()
