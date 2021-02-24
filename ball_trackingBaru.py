from collections import deque
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

orangeLower = (0, 91, 45)
orangeUpper = (37, 255, 255)

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    
    #cv2.putText(frame, ('x='+str(int(x))), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,100), 2, cv2.LINE_AA)
    #cv2.putText(frame, ('y='+str(int(y))), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,100), 2, cv2.LINE_AA)
    #cv2.putText(frame, ('z='+str(int(radius))), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,100), 2, cv2.LINE_AA)


    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

