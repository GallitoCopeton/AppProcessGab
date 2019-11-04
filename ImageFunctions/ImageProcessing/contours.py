import cv2

# Hierarchy contours, normally test in pos 5 and qr in pos 3 and 4


def findTreeContours(image, area=20000):
    listPoints = []
    _, cnts, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        # http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
        epsilon = 0.1*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if(len(approx) == 4 and cv2.contourArea(approx) > area):
            listPoints.append(approx)
    return listPoints

# Find Test Squareprin


def findExternalContours(image, area=20000):
    listPoints = []
    _, cnts, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        # http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
        epsilon = 0.1*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if(len(approx) == 4 and cv2.contourArea(approx) > area):
            listPoints.append(approx)
    return listPoints
