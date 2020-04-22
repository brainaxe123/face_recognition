import cv2
import numpy as np 
from numba import vectorize
from numba import jit
#@vectorize(['float32(float32, float32)'], target='cuda')

#Init webcam
cap=cv2.VideoCapture(0)

#Face_detection

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data=[]
dataset_path = 'D:/ml/face_detection/data/'
file_name = input('Enter name of person :')
#reading image

while True:
	
	ret , frame = cap.read()

	if ret == False:
		continue

	#Convert to gray_scale

	gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
	#face_cascade.detectMultiScale returns 4 parameters

	faces = face_cascade.detectMultiScale(gray_frame , 1.3 , 5)
	if len(faces) == 0:
		continue
	faces = sorted(faces , key = lambda f : f[2]*f[3])

	#drawing bounding box
	#choosing last face in faces as it is largest

	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(gray_frame , (x,y) , (x+w , y+h) ,  (0,255,255) , 2)
		#cropping out face
		offset=10
		face_section = gray_frame[y-offset:y+offset+h , x-offset : x + offset + w]
		face_section = cv2.resize(face_section , (100,100))
		face_data.append(face_section)
		print(len(face_section))
	#cv2.imshow("Frame",frame)
	cv2.imshow("Gray_Frame" , gray_frame)
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break
#Converting face_data to numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0] , -1))
print(face_data.shape)

#Saving_data

np.save(dataset_path + file_name + '.npy' , face_data)
print('Data saved successfully :)')




cap.release()
#cap.destroyAllWindows()



