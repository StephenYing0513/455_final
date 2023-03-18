# 455 final project: Real-time hand gesture recognition - Stephen Ying #

For my real-time hand gesture recognition project, I mainly split the work into 2 main python scrips. The first scrip collects images of different hand gestures with different meaning/labeling through webcam, and the second script is to use the model built from training our images to determine what is the hand gesture currently showed in the webcam.

## Collecting Images of Our Hand Gestures: collect_images.py ##
To use my webcam to collect images for each label, I accomplished this through OpenCv. To regoning our hand, I used cvzone's HandDetectorModule. Because the classification needs to compare the image of just our hand, I majority of my work is to determine how to crop my hand down, and I was able to do it with information returned by the detector and some math to crop the images and overlay the hand to a consistent size to prevent losses during training. Usage: press key 's' and it will start collect data and put it into the specified label directory.

Frameworks
```
pip install opencv-python
pip install cvzone
```

## Training ##
To get our model, I used the open source provided by Google teachable machine to train our images and get our model through tensorflow.keras. I have tried to train my data using the tensorflow.keras and applying CNN to classify gestures, but I cannot manage to finish it with a good accuracy.
Framework:
```
pip install tensorflow
```

## Classification & Return Result: display.py ##
Now, I can classify and return the result using our model and the classifier provided by cvzone in real time. It is basically the same idea as collect images for our hand except now we feed the image to the classifier so that it display the result. Testing results are shown in demo video.

## Limitations ##
The program can only regonize hand gesture, and any gesture involves body parts that's more than the hand, it would not work due to the limitation of the HandDetector.


####
[Demo](https://zoom.us/rec/play/AaZiS-yEtPFcDWgrb0gh57ta-KHmAAtPoT5ux3XbPfURCrUq6r_lZFJxAkQIaSmG7BYqztL-MYrftaU9.GcKvnEoyPkZC-NRk?autoplay=true&startTime=1679119050000)

