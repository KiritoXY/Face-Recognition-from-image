# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:53:53 2019

@author: Adil Ayub
"""

import face_recognition
import cv2
import numpy as np
import glob

#files=[]
known_face_encodings = []
known_face_names=[]
directory_name="D:/University/Semester 7/HCI/Project/Images/"
dir_len=len(directory_name)
temp=""
for file in glob.glob(directory_name+"*.jpg"):
    #temp=file[dir_len:len(file)]
    temp_image = face_recognition.load_image_file(file)
    temp_face_encoding = face_recognition.face_encodings(temp_image)[0]
    known_face_encodings.append(temp_face_encoding)
    temp=file[dir_len:len(file)]
    temp=temp[0:(len(temp)-4)]
    known_face_names.append(temp)

    
'''       

obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a sample picture and learn how to recognize it.
adil_image = face_recognition.load_image_file("adil.jpg")
adil_face_encoding = face_recognition.face_encodings(adil_image)[0]

# Load a sample picture and learn how to recognize it.
messi_image = face_recognition.load_image_file("messi.jpg")
messi_face_encoding = face_recognition.face_encodings(messi_image)[0]

# Load a sample picture and learn how to recognize it.
arima_image = face_recognition.load_image_file("Arima.png")
arima_face_encoding = face_recognition.face_encodings(arima_image)[0]



# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    adil_face_encoding,
    messi_face_encoding,
    arima_face_encoding
]
known_face_names = [
    "Obama",
    "Adil",
    "Messi",
    "Arima"
]


'''
#frame=open("adil.jpg","rb")
#frame=frame.read()
frame=cv2.imread("adil.jpg")
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
rgb_small_frame = small_frame[:, :, ::-1]
face_locations = face_recognition.face_locations(rgb_small_frame)
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
        print(name)
    else:
        name = "Unknown"
        print("Unknown")    
    
    # # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]
    
    # Or instead, use the known face with the smallest distance to the new face
    
