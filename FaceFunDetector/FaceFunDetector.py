import cv2
import numpy as np
import matplotlib.pyplot as mpl
import sys
import random
import datetime

use_faces = True
use_eyes = True
detect_faces = True
wants_faces = True
wants_frame = True
num_of_faces = 6
face_number = 0
speed = 1 # final speed (default is 1 * 20)
# Function to put transparent image on other image ("watermark")
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):	
	bg_img = background_img.copy()
	
	if overlay_size is not None:
		img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

	# Remove alpha (4th dimension)
	b,g,r,a = cv2.split(img_to_overlay_t)
	overlay_color = cv2.merge((b,g,r))
	
	# Remove blur on edges
	mask = cv2.medianBlur(a,5)

	h, w, _ = overlay_color.shape
	roi = bg_img[y:y + h, x:x + w]

	# to make background black
	img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
	
	# mask image
	img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

	# create new image
	bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)
	# and return it
	return bg_img





# load resources - images and cascades
l_img = cv2.imread("res/smile/banana.png", -1)
l2_img = [cv2.imread("res/faces/face1.png", -1), cv2.imread("res/faces/face2.png", -1), cv2.imread("res/faces/face3.png", -1),cv2.imread("res/faces/face4.png", -1),cv2.imread("res/faces/face5.png", -1),cv2.imread("res/faces/face6.png", -1),cv2.imread("res/faces/face7.png", -1),]
leye_img = cv2.imread("res/eyes/leye.png", -1)
reye_img = cv2.imread("res/eyes/reye.png", -1)
faceCascade = cv2.CascadeClassifier("classifiers/Classifiers.xml")
mouth_cascade = cv2.CascadeClassifier("classifiers/haarcascade_mcs_mouth.xml")
eyes_cascade = cv2.CascadeClassifier("classifiers/haarcascade_mcs_eyepair_small.xml")
left_eye_cascade = cv2.CascadeClassifier("classifiers/haarcascade_mcs_lefteye.xml")
right_eye_cascade = cv2.CascadeClassifier("classifiers/haarcascade_mcs_righteye.xml")
# -----

# prepare camera recording and codec
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# name of saved video
time_now = datetime.datetime.now().strftime("%I %M %B %d %Y") + '.avi'
out = cv2.VideoWriter(time_now,fourcc,20.0 * speed,(640,480))
while True:
	# get frame
	ret,frame = cap.read()
	image = frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	if use_faces == True:
		for (x, y, w, h) in faces: # foreach face
			if wants_faces == False: # if user doesn't want to put custom face
				if wants_frame == True: # but wants to put a frame around faces
					# draw framw with random color
					cv2.rectangle(image, (x, y), (x + w, y + h), (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 5)
			else: # user wants to put custom faces
				l3 = l2_img[face_number] # load face form array
				l3 = cv2.resize(l3, (w,h)) # resize fake face to fit orginal face
				face_number = face_number + 1
				image = overlay_transparent(image, l3, x,y, (w,h)) # put custom face on frame
				if face_number > num_of_faces: # reset number of face
					face_number = 0
			r_gray = gray[y:y+h,x:x+w]
			# detect eyes
			if use_eyes == True:
					# detect eyes in each face
					right_eye_rects = right_eye_cascade.detectMultiScale(r_gray, 1.7, 5)
					have_eyes = 0
					# foreach eye (not more than two in one face)
					for (x2,y2,w2,h2) in right_eye_rects:
						if have_eyes >= 2:
							break
						have_eyes = have_eyes + 1
						y = int(y - 0.15 * h2)
						# get eye image, resize and put it on the frame
						l_img_bkup2 = reye_img
						l_img_bkup2 = cv2.resize(l_img_bkup2, (w2,h2))
						image = overlay_transparent(image, l_img_bkup2, x2 + x,y2 + y, (w2,h2))
			# detect mouth, same logic as for eyes (except there's only one mouth for one face)
			if detect_faces == True:
				mouth_rects = mouth_cascade.detectMultiScale(r_gray, 1.7, 5)
				have_mouth = 0
				for (x3,y3,w3,h3) in mouth_rects:
					if have_mouth >= 1:
						break
					have_mouth = have_mouth + 1
					y = int(y - 0.15 * h3)
					l_img_bkup = l_img
					l_img_bkup = cv2.resize(l_img_bkup, (w3,h3))
					image = overlay_transparent(image, l_img_bkup, x3 + x,y3 + y, (w3,h3))

	
	face_number = 0
	
	
	
	# show result in a window frame
	cv2.imshow("Face Finder", image)
	out.write(image)
	# keyboard settings
	key_entered = cv2.waitKey(1) # read key (wait for a key for 1ms)
	# if key was pressed then set selected option:
	if key_entered == ord('q'):
		break
	elif key_entered == ord('f'):
		use_faces = not use_faces
	elif key_entered == ord('e'):
		use_eyes = not use_eyes
	elif key_entered == ord('0'):
		num_of_faces = 0
	elif key_entered == ord('1'):
		num_of_faces = 1
	elif key_entered == ord('2'):
		num_of_faces = 2
	elif key_entered == ord('3'):
		num_of_faces = 3
	elif key_entered == ord('4'):
		num_of_faces = 4
	elif key_entered == ord('5'):
		num_of_faces = 5
	elif key_entered == ord('6'):
		num_of_faces = 6
	elif key_entered == ord('m'):
		detect_faces = not detect_faces
	elif key_entered == ord('-'):
		wants_faces = not wants_faces
	elif key_entered == ord('='):
		wants_frame = not wants_frame