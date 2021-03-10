import sys
import cv2
import os
import numpy as np
import argparse
import imutils
import pickle
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton

class MyDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 300)
        self.label = QLabel()
        self.btnOpen = QPushButton('Open Image', self)
        self.btnProcess = QPushButton('Crop Face', self)
        self.btnSave = QPushButton('Save Face', self)
        self.btnSave.setEnabled(False)

        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 0, 4, 4)
        layout.addWidget(self.btnOpen, 4, 0, 1, 1)
        layout.addWidget(self.btnProcess, 4, 1, 1, 1)
        layout.addWidget(self.btnSave, 4, 2, 1, 1)

        self.btnOpen.clicked.connect(self.openSlot)
        self.btnProcess.clicked.connect(self.processSlot)
        self.btnSave.clicked.connect(self.saveSlot)

    def openSlot(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if filename is '':
            return
        self.img = cv2.imread(filename, -1)
        if self.img.size == 1:
            return
        self.showImage()
        self.btnSave.setEnabled(True)

    def saveSlot(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Image', 'Image', '*.png *.jpg *.bmp')
        if filename is '':
            return
        cv2.imwrite(filename, self.crop_img)

    def processSlot(self):

        # construct the argument parser and parse the arguments
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-i", "--image", required=True,
        #     help="path to input image")
        # ap.add_argument("-d", "--detector", required=True,
        #     help="path to OpenCV's deep learning face detector")
        # ap.add_argument("-m", "--embedding-model", required=True,
        #     help="path to OpenCV's deep learning face embedding model")
        # ap.add_argument("-r", "--recognizer", required=True,
        #     help="path to model trained to recognize faces")
        # ap.add_argument("-l", "--le", required=True,
        #     help="path to label encoder")
        # ap.add_argument("-c", "--confidence", type=float, default=0.5,
        #     help="minimum probability to filter weak detections")
        # args = vars(ap.parse_args())

        args = {}
        args["detector"] = "face_detection_model"
        args["recognizer"] = "output/recognizer.pickle"
        args["embedding_model"] = "openface_nn4.small2.v1.t7"
        args["le"] = "output/le.pickle"
        args["confidence"] = 0.5

        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
        modelPath = os.path.sep.join([args["detector"],
            "res10_300x300_ssd_iter_140000.caffemodel"])
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # load our serialized face embedding model from disk
        print("[INFO] loading face recognizer...")
        embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

        # load the actual face recognition model along with the label encoder
        recognizer = pickle.loads(open(args["recognizer"], "rb").read())
        le = pickle.loads(open(args["le"], "rb").read())

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image dimensions
        image = self.img
        image = imutils.resize(image, width=1000)
        (h, w) = image.shape[:2]

        # construct a blob from the image cv2.resize(image, (300, 300)), 1.0, (300, 300),
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        k=0

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            print(k)
                
            # filter out weak detections
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for the
                # face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                k = k+1
                # extract the face ROI
                face = image[startY:endY, startX:endX]
                self.crop_img = face
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 10 or fH < 10:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                    (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # draw the bounding box of the face along with the associated
                # probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    (0, 0, 255), 1)
                cv2.putText(image, name, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        self.img = image
        # self.crop_img = face
        self.showImage()

    def showImage(self):
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))

if __name__ == '__main__':
    a = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(a.exec_())