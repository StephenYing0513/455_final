import tensorflow as tf
import scipy
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

img = image.load_img("images/train/C/Image_1679095958.7013.jpg")

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

# convert training images to a dataset that can feed to our neural network
train_dataset = train.flow_from_directory('images/train/',target_size=(300,300),batch_size=3)
validation_dataset = train.flow_from_directory('images/validation/',target_size=(300,300),batch_size=3)

# print(train_dataset.class_indices)

'''model defining'''
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3), activation='relu', input_shape=(300,300,3)),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid'),
                                    ])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
model.fit = model.fit_generator(train_dataset, steps_per_epoch=5, epochs=30, validation_data=validation_dataset)


#
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# offset = 20
# imgSize = 300
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# labels = ["A", "B"]
#
#
# while True:
#     success, img = cap.read()
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
#
#     # Crop Image
#     if hands:
#         # Hand 1
#         hand1 = hands[0]
#         x, y, w, h = hand1["bbox"]  # Bounding box info x,y,w,h
#
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 # HandPicture to be collected
#
#         imgCrop = img[(y-offset):(y+h+offset), (x-offset):(x+w+offset)] # Our hand
#
#         aspectRatio = h/w
#
#         if aspectRatio > 1:
#             k = imgSize/h # how much height of our hand stretched
#             wCal = math.ceil(k*w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#
#             # calculate the center position of our hand to be in imgWhite
#             wGap = math.ceil((imgSize-wCal)/2)
#
#             imgWhite[:, wGap:wCal+wGap] = imgResize
#
#             prediction, index = classifier.getPrediction(imgWhite)
#             print(prediction, index)
#
#
#         else:
#             k = imgSize/w  # how much width of our hand stretched
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#
#             # calculate the center position of our hand to be in imgWhite
#             hGap = math.ceil((imgSize - hCal) / 2)
#
#             imgWhite[hGap:hCal + hGap:, :] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite)
#             print(prediction, index)
#
#         cv2.rectangle(imgOutput, (x - offset - 2, y - offset - 50),
#                                  (x + w + 22, y - offset), (255, 0, 255), cv2.FILLED)
#         cv2.putText(imgOutput, labels[index], (x + 25, y - offset - 5), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
#         cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
#
#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
#
#     cv2.imshow("Image", imgOutput)
#     cv2.waitKey(1)
