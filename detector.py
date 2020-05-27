import cv2

shotCount = 0

# import haar cascades
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')


def detect(gray, img):
    global shotCount

    # face detection
    face = faceCascade.detectMultiScale(
        gray, scaleFactor=1.03, minNeighbors=5, minSize=(100, 100))

    # face outline
    for (fx, fy, fw, fh) in face:
        cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (50, 50, 100), 2)

        # define Region Of Interest
        roiGray = gray[fy:fy+fh, fx:fx+fw]
        roiColor = img[fy:fy+fh, fx:fx+fw]

        # eyes detection
        eyes = eyesCascade.detectMultiScale(
            roiGray, scaleFactor=1.07, minNeighbors=15, minSize=(20, 20), maxSize=(150, 150))
        # eyes outline
        for(ex, ey, ew, eh) in eyes:
            cv2.circle(roiColor, (ex+int(ew*0.5), ey+int(eh*0.5)),
                       int(ew*0.5), (200, 50, 50), 2)

        # smile detection
        smile = smileCascade.detectMultiScale(
            roiGray, scaleFactor=1.03, minNeighbors=55, minSize=(50, 50), maxSize=(150, 150))
        # smile outline
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roiColor, (sx, sy), (sx+sw, sy+sh), (50, 200, 50), 2)

        # save frame if face and smile and eyes detected
        if(len(face) and len(smile) and len(eyes) >= 2):
            cv2.imwrite('smile{}.jpg'.format(shotCount), img)
            shotCount += 1
        print(len(face), len(smile), len(eyes), shotCount)
    return img


# capture continuous (0) video stream
capture = cv2.VideoCapture(0)

while True:
    # read the new frame from the capture stream
    ret, frame = capture.read()

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply the detector to the frame
    final = detect(gray, frame)

    #  show the captured stream
    cv2.imshow("Smile Detector", final)

    # break if 'q' character is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release video capture stream
cap.release()
cv2.destroyAllWindows()
