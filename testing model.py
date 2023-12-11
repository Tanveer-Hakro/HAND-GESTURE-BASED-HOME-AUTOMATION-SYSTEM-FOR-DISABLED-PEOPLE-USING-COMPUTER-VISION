import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import pyfirmata
import time
from pyfirmata import util, STRING_DATA


def label_decoding(encoded_label):
    label = np.argmax(encoded_label, axis=1)
    return label


port="COM9"
board=pyfirmata.Arduino(port)
cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5,max_num_hands=16)
mpDraw=mp.solutions.drawing_utils
model = load_model('LSTM_hand_gesture_model')

board.digital[7].write(1)
board.digital[6].write(1)
board.digital[5].write(1)
board.digital[4].write(1)

# font
font = cv2.FONT_HERSHEY_SIMPLEX
height=960
width=1280


# org
org = (20, 90)

# fontScale
fontScale = 3

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

data=[]
total_frame_samples=10
count_data_frame=0
land_mark_missing_rate=0
label=""
appliances=0
while True:
    success, img1 =cap.read()
    img2=img1


    # img=cv2.flip(img, 1)
    # img=cv2.resize(img, (width, height))
    # print(img.shape)
    imgRGB1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    imgRGB2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    result=hands.process(imgRGB2)
    results=hands.process(imgRGB1)

    # for window 2

    if results.multi_hand_landmarks:
       count_data_frame += 1
       for hand in results.multi_hand_landmarks:

           example = []
           for id, lm in enumerate(hand.landmark):
               example.append(lm.x)
               example.append(lm.y)


           data.append(example)
           example=np.array([example])


    else:
        land_mark_missing_rate+=1
        if land_mark_missing_rate>3:
            land_mark_missing_rate = 0
            count_data_frame = 0
            data=[]
        # print("Missing Rate",land_mark_missing_rate)

    if count_data_frame==total_frame_samples:
        data=np.array(data)
        data= np.reshape(data, (data.shape[0], 1, data.shape[1]))
        vote={1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
        predictions=model.predict(data)
        predictions=label_decoding(predictions)
        for pred in predictions:
           vote[pred]+=1
        find_max=max(vote,key=vote.get)
        appliances=find_max
        label_classes = {1:'Light1_ON' , 2 :'Light1_OFF' , 3:'Light2_ON', 4:'Light2_OFF', 5:'Light3_ON',
                         6:'Light3_OFF',7:'Fan_ON',8:'Fan_OFF', 9:'Wrong_gestures'}
        label=label_classes[find_max]
        img1 = cv2.putText(img1, label, org, font, fontScale, color, thickness, cv2.LINE_AA)
        print(find_max)
        print(vote)

        count_data_frame=0
        land_mark_missing_rate=0
        data=[]


    cv2.imshow("Image1", img1)
    img2 = np.zeros([480, 640, 3], dtype=np.uint8)
    img2.fill(255)
    if result.multi_hand_landmarks:
       for hand in result.multi_hand_landmarks:
           mpDraw.draw_landmarks(img2, hand, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image2", img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(r'C:\Users\zkh\Desktop\myproject\17.jpeg', img2)
        cv2.imwrite(r'C:\Users\zkh\Desktop\myproject\18.jpeg', img1)
        break

    cv2.waitKey(1)

    if appliances == 1:
       board.digital[7].write(0)
       board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("Light 1 ON"))
       print("light1_ON")

    elif appliances == 2:
      board.digital[7].write(1)
      board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("Light 1 OFF"))
      print("light1_OFF")

    elif appliances == 3:
       board.digital[6].write(0)
       board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("Light 2 ON"))
       print("light2_ON")

    elif appliances == 4:
      board.digital[6].write(1)
      board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("Light 2 OFF"))
      print("light2_OFF")

    elif appliances == 5:
       board.digital[5].write(0)
       board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("Light 3 ON"))
       print("light3_ON")

    elif appliances == 6:
      board.digital[5].write(1)
      board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("Light 3 OFF"))
      print("light3_OFF")

    elif appliances == 7:
        board.digital[4].write(0)
        board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("Fan ON"))

    elif appliances == 8:
        board.digital[4].write(1)
        board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("Fan OFF"))









""""
for hardware part
if find_max==1:	
	for light 1
else if find_max==2:
	for light 1 off
else if find_max==3:
	for light2 on
else if find_max==4:
	for light2 off
else if find_max==5:
	for light3 on
else if find_max==6:
	for light3 off
else if find_max==7:
	 for Fan on
else if find_max==8:
         for Fan off
"""

