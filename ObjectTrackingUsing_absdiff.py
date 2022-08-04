import cv2

cap = cv2.VideoCapture('highway_cars.mp4')
vehicle = 0

success, frame1 = cap.read()
frame1 = cv2.resize(frame1, (640, 480))

while True:

    success, frame2 = cap.read()

    # check if end of frame is reached
    if not success:
        break
    frame2 = cv2.resize(frame2, (640, 480))
    frame = frame2.copy()

    # Extract the foreground mask
    fgMask = cv2.absdiff(frame1, frame2)

    # convert foreground into GRAY
    fgMask = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)

    # apply the threshold for increasing white foreground
    _, thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)

    # assign frame2 to frame1 to continue the iteration untill all frames are read.
    frame1 = frame2

    # draw the reference traffic lines
    cv2.line(frame, (155, 272), (428, 272), (0, 255, 0), 1) # Green offset above
    cv2.line(frame, (132, 295), (444, 295), (0, 0, 255), 2)
    cv2.line(frame, (117, 324), (461, 324), (0, 255, 0), 1) # Green offset below

    # extract the contours
    conts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in conts:
        if cv2.contourArea(c) < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)

        # draw the bounding rectangle for all contours
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
        xMid = int((x+(x+w))/2)
        yMid = int((y+(y+h))/2)
        cv2.circle(frame, (xMid, yMid), 5, (0,0,255), 2)

        if yMid > 270 and yMid < 324:
            vehicle += 1


    # show the thresh and original image
    cv2.imshow('Foreground mask', thresh)

    cv2.putText(frame, f'Total vehicle : {vehicle}', (150, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    cv2.imshow('Original Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()