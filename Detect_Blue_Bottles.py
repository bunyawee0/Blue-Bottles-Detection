import cv2
import numpy as np
import os

def readImage(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (1279, 641))
    return img

def detectionBlueBottles(path):
    img = readImage(path)
    img_copy = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img, 3)

    upper_blue = np.array([130, 255, 255])
    lower_blue = np.array([90, 0, 0])
    
    canny = cv2.Canny(img_blur, 100, 200)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    h, w = img.shape

    mindist = int(min(h, w) / 20)
    param1 = int(np.mean(canny) * 0.5) * 6.5
    param2 = int(np.mean(canny) / 2) * 6.5
    minradius = int(min(h, w) / 40 * 0.5)
    maxradius = int(min(h, w) / 40 * 0.9)

    count_bottles = []
  
    if mask is not None:
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist= mindist, param1= param1, param2=param2, minRadius=minradius, maxRadius=maxradius)
        if circles is not None:
            circles = np.uint16(np.around(circles))

            for point in circles[0]:
                x, y, r = point
                cv2.circle(mask, (x ,y), r, (0, 255, 0), 2)
        
        blue_bottles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, minDist= mindist, param1= param1, param2=param2, minRadius=minradius, maxRadius=maxradius)
        if blue_bottles is not None:
            blue_bottles = np.uint16(np.around(blue_bottles))
            
            for point in blue_bottles[0]:
                x, y, r = point
                cv2.circle(img_copy, (x ,y), r, (0, 255, 0), 2)
                count_bottles.append(point)
    
        else:
            print("No blue bottles detected")

    while True:
        cv2.putText(img_copy, "Total:"+ str(len(count_bottles)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Result", img_copy)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

            cv2.destroyAllWindows()

    return count_bottles


if __name__ == "__main__":
    path = 'Data_Bottles (1).png'
    # path = "Example_Test_Bottles (1).png"
    detectionBlueBottles(path)
