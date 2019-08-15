import cv2
import os


os.chdir("C:\melfaiz\eigenfaces")
from main import *

cam = cv2.VideoCapture(0)

[train,tr_labels,shape] = load_images("C:\melfaiz\eigenfaces\\faces\\train")

E,V = pca(train)

while True:



    ret, frame = cam.read()

    #flip the frame vertically
    frame = cv2.flip( frame, 1 )

    #gray frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #haar cascade detection
    face_cascade = cv2.CascadeClassifier("C:\melfaiz\eigenfaces\\haarcascade_frontalface_alt.xml")
    face_detect = face_cascade.detectMultiScale(gray_frame,scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h) in face_detect:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        crop_img = gray_frame[y+2:y+h-2, x+2:x+w-2]

        crop_img = cv2.resize(crop_img,(shape[0],shape[1]))
        flat_img = crop_img.flatten()

        #show face namee

        font = cv2.FONT_HERSHEY_SIMPLEX
        name,error  = image_predict(flat_img,train,E,tr_labels)
        acc = (1-error/27000)*100

        cv2.putText(frame, name + " " + str(int(abs(acc))) + "%" , (x, y+h+20), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("face detection", frame)

    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27 or cv2.getWindowProperty('face detection', 0) < 0 :
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        print(image_predict(flat_img,E))

cam.release()

cv2.destroyAllWindows()