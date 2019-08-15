import cv2
import matplotlib.pylab as plt
import sys
import os
from PIL import Image

base = "C:\melfaiz\eigenfaces\\faces"

def new_face(name,path):

    path = os.path.join(base, path)
    os.chdir(path)

    cam = cv2.VideoCapture(0)

    img_counter = 0

    images = []

    shape = (100,100)

    w = int(cam.get(3))  # float
    h = int(cam.get(4)) # float


    H,W=[],[]

    while True:

        ret, frame = cam.read()

        #flip the frame vertically
        frame = cv2.flip( frame, 1 )

        #gray frame

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #show captures number
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Capture "+str(img_counter), (240, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        #haar cascade detection
        face_cascade = cv2.CascadeClassifier("C:\melfaiz\eigenfaces\\haarcascade_frontalface_alt.xml")
        face_detect = face_cascade.detectMultiScale(gray_frame,scaleFactor=1.1, minNeighbors=5)

        for (x,y,w,h) in face_detect:
            H,W=H+[h],W+[w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            crop_img = frame[y+2:y+h-2, x+2:x+w-2]
            crop_img = cv2.resize(crop_img,(shape[0],shape[1]))
        #show frame
        cv2.imshow("training", frame)

        if not ret:
            break

        k = cv2.waitKey(1)

        if k%256 == 27 or cv2.getWindowProperty('training', 0) < 0:
            # ESC pressed
            print("Escape hit, closing...")
            break

        else :
            if k%256 == 32:

                img_name = "opencv_frame_{}.png".format(img_counter)

                images.append(crop_img)
                title = name+ " " + str(img_counter)+".png"
                cv2.imwrite(title,crop_img)
                img_counter += 1



    cam.release()

    cv2.destroyAllWindows()
