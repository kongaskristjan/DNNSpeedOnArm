
import cv2
import numpy as np

class OpenCVRunner:
    def __init__(self, modelPath):
        self.net = cv2.dnn.readNetFromTensorflow(modelPath)

    def inference(self, input):
        blob = cv2.dnn.blobFromImage(input, scalefactor=1./255., swapRB=False)
        self.net.setInput(blob)
        output = self.net.forward()
        return output
