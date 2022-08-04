# Extract the foreground mask from the background

import cv2
cap = cv2.VideoCapture('highway_cars.mp4')
vehicle = 0

# initilize OpenCV - Background subtractor for KNN and MOG2
BS_KNN = cv2.createBackgroundSubtractorKNN()
BS_MOG2 = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (640, 480))

    # extract the KNN-method of Foreground Mask
    #fgMask = BS_KNN.apply(frame)

    # extract the MOG2-method of Foreground Mask
    fgMask = BS_MOG2.apply(frame)

    # draw the reference traffic lines
    cv2.line(frame, (155, 272), (428, 272), (0, 255, 0), 1) # Green offset above
    cv2.line(frame, (132, 295), (444, 295), (0, 0, 255), 2)
    cv2.line(frame, (117, 324), (461, 324), (0, 255, 0), 1) # Green offset below

    # extract the contours
    conts, _ = cv2.findContours(fgMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in conts:
        if cv2.contourArea(c) < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if x > 155 and x < 500 and y > 100:
            # draw the bounding rectangle for all contours
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
            xMid = int((x+(x+w))/2)
            yMid = int((y+(y+h))/2)
            cv2.circle(frame, (xMid, yMid), 5, (0,0,255), 2)

            if yMid > 270 and yMid < 324:
                vehicle += 1


    # show the thresh and original image
    cv2.imshow('Foreground mask', fgMask)

    cv2.putText(frame, f'Total vehicle : {vehicle}', (150, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    cv2.imshow('Original Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()