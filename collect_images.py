import math
import cv2
import numpy as np
import time
import os
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

folder = "images/C"
counter = 0

os.mkdir(folder)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    # Crop Image
    if hands:
        # Hand 1
        hand1 = hands[0]
        x, y, w, h = hand1["bbox"]  # Bounding box info x,y,w,h

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 # HandPicture to be collected

        imgCrop = img[(y-offset):(y+h+offset), (x-offset):(x+w+offset)] # Our hand

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h # how much height of our hand stretched
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape

            # calculate the center position of our hand to be in imgWhite
            wGap = math.ceil((imgSize-wCal)/2)

            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
            k = imgSize / w  # how much height of our hand stretched
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            # calculate the center position of our hand to be in imgWhite
            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[hGap:hCal + hGap:, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)