# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

from adafruit_servokit import ServoKit 

def inRange(center, range, compareValue):
	return ((center - range) < compareValue) or (compareValue < (center + range))

def fixToWindowsCenter(servoKit, WINDOWS_SIZE, faceCenterX, faceCenterY, errorAccpet = 20, FIX_ANGLE = 3):
	WindowsCenterX, WindowsCenterY  = WINDOWS_SIZE / 2, WINDOWS_SIZE / 2
	Xservo = 0
	Yservo = 1
	inCenter = True
	#fix X
	if (not inRange(WindowsCenterX, errorAccpet, faceCenterX)):
		inCenter = False
		#face too left
		if (WindowsCenterX > faceCenterX):
			print("Too Left! Turn camara LEFT")
			if(servoKit.servo[Xservo].angle - FIX_ANGLE <= 0):
				servoKit.servo[Xservo].angle = 0
				print("Servo X angle = 0, Can't rotate more.")
			else:
				servoKit.servo[Xservo].angle = servoKit.servo[Xservo].angle - FIX_ANGLE

		#face too right
		if (WindowsCenterX < faceCenterX):
			print("Too Right, Turn camara RIGHT")
			if(servoKit.servo[Xservo].angle + FIX_ANGLE >= 180):
				servoKit.servo[Xservo].angle = 180
				print("Servo X angle = 180, Can't rotate more.")
			else:
				servoKit.servo[Xservo].angle = servoKit.servo[Xservo].angle + FIX_ANGLE

	#fix Y
	if (not inRange(WindowsCenterY, errorAccpet, faceCenterY)):
		inCenter = False
		#face too low
		if (WindowsCenterY > faceCenterY):
			print("Face too low! Turn camara DOWN")
			if(servoKit.servo[Yservo].angle - FIX_ANGLE <= 0):
				servoKit.servo[Yservo].angle = 0
				print("Servo Y angle = 0, Can't rotate more.")
			else:
				servoKit.servo[Yservo].angle = servoKit.servo[Yservo].angle - FIX_ANGLE

		#face too high
		if (WindowsCenterX < faceCenterX):
			print("Face too high, Turn camara UP")
			if(servoKit.servo[Yservo].angle + FIX_ANGLE >= 180):
				servoKit.servo[Yservo].angle = 180
				print("Servo Y angle = 180, Can't rotate more.")
			else:	
				servoKit.servo[Yservo].angle = servoKit.servo[Yservo].angle + FIX_ANGLE

	return inCenter


#Main

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

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

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


#init Servo
print("Init ServoKit")
servoKit = ServoKit(channels=16, i2c_bus=0)
servoKit.servo[0].angle = 90
servoKit.servo[1].angle = 90
# start the FPS throughput estimator
fps = FPS().start()

WINDOWS_SIZE = 600
faceCenterX = 0
faceCenterY = 0
# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			print("detect face!!")
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			faceCenterX = (endX - startX) / 2 + startX
			faceCenterY = (endY - startY) / 2 + startY

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# update the FPS counter
	fps.update()
	if( fixToWindowsCenter(servoKit, WINDOWS_SIZE, faceCenterX, faceCenterY) ):
		print("face in center")

	# show the output frame
	# cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()



