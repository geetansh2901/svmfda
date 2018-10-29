import cv2
import dlib                      # for face detection and facial landmarks detection
import numpy as np
from imutils import face_utils   # for use of two fns 1.rect_bb() 2.shape_np
import imutils                   # for easy use of open cv image processing function

# initializing dlib's pre-trained face detection method
detector = dlib.get_frontal_face_detector()
# and pre-trained shape predictor method and providing path to .dat predictor file
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

image = cv2.imread('F:\pythontraining\headPose.jpg')
image = imutils.resize(image,width=500)    # resizing image for fiting in screen  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray,1)       # detecting face 

for(i, rect) in enumerate(rects):
    shape = predictor(gray, rect)                             # predicts facial landmarks in each face in each iteration
    shape = face_utils.shape_to_np(shape)                     # converting landmark list to  a numpy array
    (x, y, w, h) = face_utils.rect_to_bb(rect)                # converting face bounding box to open cv 4 tuple rectangle 
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # creating bounding box around detected face
    for(x, y)in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)         # creating dots for facial landmarks


cv2.imshow("IMAGE",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
