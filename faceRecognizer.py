import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from flask import Flask, render_template, Response, request
# Path to the folder containing images
path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# Load images and class names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(attendanceNames):
    df = pd.DataFrame(attendanceNames, columns=['Name'])
    cur_date = datetime.now().strftime('%Y-%m-%d')
    df['Time'] = datetime.now().strftime('%H:%M')
    filename = f'Attendance_{cur_date}.csv'
    df.to_csv(filename, index=False)
    print(f"Attendance saved to {filename}")

# Find encodings for known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Video capture from file or webcam
#videopath = "C:\\Users\\vivek\\OneDrive\\Pictures\\Camera Roll\\WIN_20240711_09_38_11_Pro.mp4"
videopath=0
cap = cv2.VideoCapture(videopath)
if not cap.isOpened():
    print("Could not find video source")
    exit()

# Initialize attendance list and thread pool executor
attendanceNames = []
executor = ThreadPoolExecutor(max_workers=2)

def process_frame(frame, encodeListKnown, classNames):
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    detected_names = []
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            detected_names.append(name)
    return frame, detected_names

while True:
    success, img = cap.read()
    if not success:
        print("Capture failed")
        break

    # Submit the frame to the thread pool for processing
    future = executor.submit(process_frame, img, encodeListKnown, classNames)
    img, detected_names = future.result()

    # Update attendance list with detected names
    for name in detected_names:
        if name not in attendanceNames:
            attendanceNames.append(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown()

markAttendance(attendanceNames)
