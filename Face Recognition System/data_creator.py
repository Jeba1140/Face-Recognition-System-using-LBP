import numpy as np
import cv2
import sqlite3

camera = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

roll = input("Enter the roll number: ")
name = input("Enter the name: ")


i=0
while True:
     ret, frame = camera.read()
     gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     face = classifier.detectMultiScale(gray_frame,1.3,5)
     for x,y,w,h in face:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),5)
          cv2.imshow("myface", frame)
          cv2.imwrite("facedata/students."+str(roll)+"."+str(i)+".jpg",gray_frame[y:y+h,x:x+w])
          i = i+1
     if cv2.waitKey(1) & 0xFF == ord('q'):
          break
     elif i>100:
          break

camera.release()
cv2.destroyAllWindows()
     
     
















     

