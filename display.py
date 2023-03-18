import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["A", "B"]


while True:
    success, img = cap.read()
    imgOutput = img.copy()
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

            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)


        else:
            k = imgSize/w  # how much width of our hand stretched
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            # calculate the center position of our hand to be in imgWhite
            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[hGap:hCal + hGap:, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)

        cv2.rectangle(imgOutput, (x - offset - 2, y - offset - 50),
                                 (x + w + 22, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x + 25, y - offset - 5), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
