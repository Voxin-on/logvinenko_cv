import cv2
import numpy as np
import time
from math import dist
import json
from pathlib import Path

import random

save_path = Path(__file__).parent

cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

clicked = False
def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at {x}, {y}")
        global position
        global clicked
        position = [x, y]
        clicked = True
        
cv2.setMouseCallback("Image", on_click)
capture=cv2.VideoCapture(0+cv2.CAP_DSHOW)

position = []
trackers = [{"lower": None, "upper": None} 
            for _ in range(4)]
active = 0

config_path = save_path / "config.json"
if config_path.exists():
    with config_path.open("r") as f:
        js = json.load(f)
        for i, tracker in enumerate(js.get("trackers", [])):
            if tracker["lower"] is not None:
                trackers[i]["lower"] = np.array(tracker["lower"], dtype="u1")
                trackers[i]["upper"] = np.array(tracker["upper"], dtype="u1")

prev_time = time.time()
curr_time = time.time()
d = 6.36 #cm

secret = random.sample([0, 1, 2], 3)

check = "none"

while True:
    key = cv2.waitKey(50) & 0xFF
    if key == ord("n"): check = 'new_three'
    if key == ord("m"): check = 'new_four'
    
    if check == 'new_three':
        secret = random.sample([0, 1, 2], 3)
        print(secret)
        check = "none"
        
    if check == 'new_four':
        secret = random.sample([0, 1, 2, 3], 4)
        print(secret)
        check = "none"
        
    ret, frame = capture.read()
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    if key == ord("q"):
        break
    if key == ord("1"): active = 0
    if key == ord("2"): active = 1
    if key == ord("3"): active = 2
    if key == ord("4"): active = 3
    if clicked:
        clicked = False
        color = hsv[position[1], position[0]]
        trackers[active]["lower"] = np.clip(color * 0.9, 0, 255).astype("u1")
        trackers[active]["upper"]  = np.clip(color * 1.1, 0, 255).astype("u1")
        trackers[active]["upper"][1] = 255
        trackers[active]["upper"][2] = 255
        
    found = {}
    
    for i, tracker in enumerate(trackers):
        if tracker["lower"] is None:
            continue
        inr = cv2.inRange(hsv, tracker["lower"], tracker["upper"])
        mask = cv2.morphologyEx(inr, cv2.MORPH_CLOSE,
                            np.ones((5,5),dtype="u1"))
        cv2.imshow("Mask",mask)
        contours, _ =cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                found[i] = (int(x), int(y))   
                cv2.circle(frame, (int(x), int(y)), 
                        int(radius), (0, 255, 255), 4)
                cv2.circle(frame, (int(x), int(y)), 5,
                        (0, 255, 255), -1)
    cv2.imshow("Image",frame)
    
    if len(found) == 3:
        order = [idx for idx, _ in sorted(found.items(), key=lambda item: item[1][0])]
        # print(order)
        
        if order == secret:
            print("отгадал")
            
    if len(found) == 4:
        order_x = [idx for idx, _ in sorted(found.items(), key=lambda item: item[1][0])]
        order_y = [idx for idx, _ in sorted(found.items(), key=lambda item: item[1][1])]
        print(order)
        
        if order_x == secret or order_y == secret:
            print("отгадал")
        
with (save_path / "config.json").open("w") as f:
    json.dump({
        "trackers":[{
        "lower": None if tracker["lower"] is None else tracker["lower"].tolist(),
        "upper": None if tracker["lower"] is None else tracker["upper"].tolist()
        }
        for tracker in trackers
        ]
    },f)