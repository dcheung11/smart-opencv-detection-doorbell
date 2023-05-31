import cv2
import numpy as np
from twilio.rest import Client
import time
# Twilio account credentials
account_sid = 'AC4ac1ca07d143a1035f6edac3d03ee42e'
auth_token = '94c08e8db9d2d96666ca06045f85e66c'
client = Client(account_sid, auth_token)

# Twilio phone number and recipient's phone number
twilio_phone_number = '+13204464421'  # Your Twilio phone number
recipient_phone_number = '+17057680341'  # Recipient's phone number

# open cv dnn
net = cv2.dnn.readNet("./dnn_model/yolov4-tiny.weights","./dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

# init camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)




classes = []
with open("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name=class_name.strip()
        classes.append(class_name)

# create window

# cv2.namedWindow("Frame")
# cv2.setMouseCallback("Frame", click_button)


# Define cooldown period in seconds
cooldown_period = 60  # Set the desired cooldown period here (e.g., 60 seconds)

dog_present = False
last_alert_time = time.time() - cooldown_period  # Initialize with a time earlier than the cooldown period

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
            print("yep")
            # check if cooldown period has elapsed since the last alert

            dog_present = True
            cv2.putText(frame, "Dog", (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # check if a dog is present and send an SMS alert
    if dog_present:
        current_time = time.time()
        if current_time - last_alert_time >= cooldown_period:

            # send SMS alert
            message = client.messages.create(
                body='A dog has been detected!',
                from_=twilio_phone_number,
                to=recipient_phone_number
            )
            print("SMS alert sent.")
            
            # update last alert time
            last_alert_time = time.time()

        # reset dog_present flag after alerting
        dog_present = False

    cv2.imshow("frame", frame)
    cv2.waitKey(1)
