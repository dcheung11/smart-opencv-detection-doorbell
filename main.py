import cv2
import numpy as np
from twilio.rest import Client
import time
import pygame

# Get audio path 
alarm_path = "./alarm.wav"

# Pygame setup for audio notifications
pygame.mixer.init()
speaker_volume = 0.99 #99% volume
pygame.mixer.music.set_volume(speaker_volume)
pygame.mixer.music.load(alarm_path)

# Twilio account credentials - Replace with your credentials
account_sid = 'YOUR_ACCOUNT_SID'
auth_token = 'YOUR_AUTH_TOKEN'
client = Client(account_sid, auth_token)

# # Twilio phone number and recipient's phone number
twilio_phone_number = 'YOUR_TWILIO_NUM'  # Your Twilio phone number
recipient_phone_number = 'YOUR_RECIPIENT_NUM'  # Recipient's phone number


# Open CV DNN Model Init
net = cv2.dnn.readNet("./dnn_model/yolov4-tiny.weights","./dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

# init camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

# Init detectable classes
classes = []
with open("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name=class_name.strip()
        classes.append(class_name)

# Define cooldown period in seconds
cooldown_period = 60  # Set the desired cooldown period here (e.g., 60 seconds)

# Initialize with a time earlier than the cooldown period
dog_present = False
last_alert_time = time.time() - cooldown_period  


while True:
    # get frames
    ret, frame = cap.read()

    # object detection
    (class_ids, scores, bboxes)  = model.detect(frame)

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        class_name = classes[class_id]
        (x, y, w, h) = bbox

        # check if the detected object is a dog
        if class_name == "dog":
            print("Dog detected in frame")
            dog_present = True
            cv2.putText(frame, "Dog", (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # check if a dog is present and send an SMS alert and play doorbell sound
    if dog_present:
        current_time = time.time()
        if current_time - last_alert_time >= cooldown_period:
            pygame.mixer.music.play()
            # send SMS alert
            message = client.messages.create(
                body='A dog has been detected - go check the door!',
                from_=twilio_phone_number,
                to=recipient_phone_number
            )
            print("SMS alert sent.")
            
            # update last alert time
            last_alert_time = time.time()

        # reset dog_present flag after alerting
        dog_present = False
    
    # Detector for my dog Saki
    cv2.imshow("SAKI DETECTOR", frame)
    cv2.waitKey(1)
