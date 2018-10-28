from scipy.spatial import distance as dist
import cv2
import imutils
from imutils import face_utils
import numpy as np
import dlib
from math import sqrt
from tkinter import * 


window = Tk()
window.title('Smart Vending Machine Software ')

but1 = Button(window, text='Withdraw', width=40, height=10,bg='#7cfc00', fg='black', bd=5)
but1.grid(row=1, column=1)
but2 = Button(window, text='Change Pin', width=40, height=10,
              bg='#7cfc00', fg='black', bd=5)
but2.grid(row=1, column=3)
but3 = Button(window, text='View Balance', width=40, height=10,
              bg='#7cfc00', fg='black', bd=5)
but3.grid(row=3, column=1)

# logo = PhotoImage(file='me.jpg')
# w1 = Label(window,image=logo).grid(row=2,column=2)

but4 = Button(window, text='Cancel Transaction', width=40, height=10,
              bg='#7cfc00', fg='black', bd=5)
but4.grid(row=3, column=3)

window.mainloop()

def EAR(eye):
    # distance between 2 set if vertical landmarks (x,y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # horizontal distance
    C = dist.euclidean(eye[0], eye[3])

    # computing EAR
    ear = (A+B)/(2.0*C)
    return ear


# defining 2 constants first as a Threshold EAR
# other for number of consecutive frames the eye must be beliow the threshold
EAR_thresh = 0.1899
EAR_consec_frames = 2

counter = 0
total = 0

# dlib's face detector
detector = dlib.get_frontal_face_detector()\
    # dlib's facial landmarks detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# geting the landmarks of lef and right eye resp.
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

faceCascade = cv2.CascadeClassifier('F:\pythontraining\projt\Head_movement_tracking\haarcascade_frontalface_alt.xml')
NOSE_POINTS = list(range(30,31))  
RIGHT_EYE_POINTS = list(range(36,37))
LEFT_EYE_POINTS = list(range(45,46))  
LEFT_LIP_POINTS = list(range(5,6))  
RIGHT_LIP_POINTS = list(range(11,12))  

cap = cv2.VideoCapture(0)


while True:
    check,frame = cap.read()
   # frame = imutils.resize(frame, width=400)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0) 
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = EAR(leftEye)
        rightEAR = EAR(rightEye)

        #average EAR
        ear = (leftEAR + rightEAR)/2.0
        #print(ear)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EAR_thresh:
            counter += 1
        else:

            if counter >= EAR_consec_frames:
                total += 1
                print(total)
                counter = 0
        cv2.putText(frame, "Blinks: {}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100),flags=cv2.CASCADE_SCALE_IMAGE  )
   
 # Draw a rectangle around the faces  
    for (x, y, w, h) in faces:  
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
   
   # Converting the OpenCV rectangle coordinates to Dlib rectangle  
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))  
   
        landmarks = np.matrix([[p.x, p.y]  
               for p in predictor(frame, dlib_rect).parts()])  
        bottomleft=(landmarks[LEFT_LIP_POINTS]-landmarks[NOSE_POINTS])
        bl= sqrt(bottomleft[0,0]*bottomleft[0,0]+bottomleft[0,1]*bottomleft[0,1])
        bottomright=(landmarks[RIGHT_LIP_POINTS]-landmarks[NOSE_POINTS])
        br= sqrt(bottomright[0,0]*bottomright[0,0]+bottomright[0,1]*bottomright[0,1])
        topleft=(landmarks[LEFT_EYE_POINTS]-landmarks[NOSE_POINTS])
        tl= sqrt(topleft[0,0]*topleft[0,0]+topleft[0,1]*topleft[0,1])
        topright=(landmarks[RIGHT_EYE_POINTS]-landmarks[NOSE_POINTS])
        tr= sqrt(topright[0,0]*topright[0,0]+topright[0,1]*topright[0,1])
        if bl<=tl and bl<=br and bl<=tr:
            landmarks_display = landmarks[LEFT_LIP_POINTS]
        elif tl<=tr and tl<=bl and tl<=br:
            landmarks_display = landmarks[LEFT_EYE_POINTS]
        elif br<=tr and br<=tl and br<=bl:
            landmarks_display = landmarks[RIGHT_LIP_POINTS]
        elif tr<=tl and tr<=br and tr<=bl:
            landmarks_display = landmarks[RIGHT_EYE_POINTS]  
        
        for idx, point in enumerate(landmarks_display):  
            pos = (point[0, 0], point[0, 1])  
            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1) 
        
    window = Tk()
    window.title('Smart Vending Machine Software ')

    but1 = Button(window, text='7', width=40, height=10,bg='#7cfc00', fg='black', bd=5)
    but1.grid(row=1, column=1)
    but2 = Button(window, text='7', width=40, height=10,
              bg='#7cfc00', fg='black', bd=5)
    but2.grid(row=1, column=3)
    but3 = Button(window, text='7', width=40, height=10,
              bg='#7cfc00', fg='black', bd=5)
    but3.grid(row=3, column=1)

    logo = PhotoImage(file=frame)
    w1 = Label(window,image=logo).grid(row=2,column=2)

    but4 = Button(window, text='7', width=40, height=10,
              bg='#7cfc00', fg='black', bd=5)
    but4.grid(row=3, column=3)

    window.mainloop() 
        
        
    cv2.imshow('Capturing',frame)
    k = cv2.waitKey(1)
    
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()