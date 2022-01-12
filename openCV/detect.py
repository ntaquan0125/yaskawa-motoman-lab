import cv2
import numpy as np
import imutils


CONVEYOR_UPPER_BOUND = 80
CONVEYOR_LOWER_BOUND = 320
CONTOUR_AREA_THRESH = 5000


def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 180)
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges


if __name__ == '__main__':
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print('Cannot open camera')
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[CONVEYOR_UPPER_BOUND:CONVEYOR_LOWER_BOUND, :]
        edges = preprocessing(frame)
        contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        try:
            screenCnt = None
            for c in contours:
                area = cv2.contourArea(c)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.015 * peri, True)
                if len(approx) and area > CONTOUR_AREA_THRESH:
                    screenCnt = approx
                    break

            hull = cv2.convexHull(screenCnt)
            M = cv2.moments(hull)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # rect = cv2.minAreaRect(screenCnt)
            # (cX, cY), (w, h), yaw = rect
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            cv2.circle(frame, (int(cX), int(cY)), 5, (255, 255, 255), -1)
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 3)
        except:
            pass

        cv2.imshow('frame', frame)
        cv2.imshow('edges', edges)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()