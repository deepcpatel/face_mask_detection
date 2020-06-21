import os
import cv2
import torch
import numpy as np
from torch import nn
from model import mask_detector
import torch.nn.functional as F

class face_mask_detector_webcam():
    def __init__(self, model_dir):
        super(face_mask_detector_webcam, self).__init__()

        self.face_det_proto = os.path.join(model_dir, "OpenCV_Face_Detector", "deploy.prototxt.txt")                        # OpenCV Face Detector Prototext
        self.face_det_model = os.path.join(model_dir, "OpenCV_Face_Detector", "res10_300x300_ssd_iter_140000.caffemodel")   # OpenCV Face Detector Trained Model
        self.mask_det_model = os.path.join(model_dir, "Mask_Detection", "mask_predictor_5.pth.tar")  # Path to Mask Detector Trained model
        self.img_resize_size = (100, 100)   # Image Resize Size
        self.string = {0: ('No Mask', (10, 0, 255)), 1: ('Masked Face', (10, 255, 0))}  # Output strings. Labels -> 0: Unmasked, 1: Masked
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Selecting Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initilizing and loading Face Detector Model
        self.confidenceThreshold = 0.6
        self.face_det_model = cv2.dnn.readNetFromCaffe(str(self.face_det_proto), str(self.face_det_model))

        # Initilizing and loading Mask Detector Model
        self.ml_model = mask_detector().to(self.device)
        self.load_ml_model(self.mask_det_model)

    def load_ml_model(self, path):       # Loads trained classifier model
        checkpoint = torch.load(path, map_location=self.device)
        # self.ml_model.load_state_dict(checkpoint['state_dict'])
        self.ml_model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.ml_model.eval()

    def detect_face(self, image):        # Detects face using OpenCV model (Referenced from the work of GitHub user JadHADDAD92)
        height, width = image.shape[0], image.shape[1]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.face_det_model.setInput(blob)
        detections = self.face_det_model.forward()
        faces = []
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidenceThreshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            startX, startY, endX, endY = box.astype("int")
            faces.append(np.array([startX, startY, endX-startX, endY-startY]))
        return faces

    def detect_mask(self):          # Fetches Webcam frames and detects mask in that
        cam = cv2.VideoCapture(2)   # Capturing from camera
        print("Press 'Esc' to quit.")

        while True:
            _, img = cam.read()
            faces = self.detect_face(img)   # Detecting Face

            for face in faces:
                xStart, yStart, width, height = face

                # clamp coordinates that are outside of the image
                xStart, yStart = max(xStart, 0), max(yStart, 0)
                
                # predict mask label on extracted face
                faceImg = img[yStart:yStart+height, xStart:xStart+width]

                # Preprocessing
                faceImg = cv2.cvtColor(cv2.resize(faceImg, self.img_resize_size), cv2.COLOR_BGR2RGB).astype(np.float32)
                faceImgTensor = torch.tensor(faceImg, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)   # NCHW format
                
                ml_pred = self.ml_model(faceImgTensor)
                
                out = torch.argmax(ml_pred, dim=1).data.numpy()             # Network output
                mask_status, color = self.string[out.astype(np.int32)[0]]   # Selecting label

                # Writing information on the frame
                cv2.rectangle(img, (xStart, yStart), (xStart + width, yStart + height), (126, 65, 64), thickness=2)
            
                # Center text according to the face frame
                textSize = cv2.getTextSize(mask_status, self.font, 1, 2)[0]
                textX = xStart + width // 2 - textSize[0] // 2
                
                # draw prediction label
                cv2.putText(img, mask_status, (textX, yStart-20), self.font, 1, color, 2)
            
            cv2.imshow('my webcam', img)

            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        cv2.destroyAllWindows()

if __name__ == '__main__':
    model_dir = './saved_models'
    detector = face_mask_detector_webcam(model_dir)
    detector.detect_mask()