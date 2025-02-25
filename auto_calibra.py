import cv2
import numpy as np

def imgSplit():
    yellow_hsv_min = np.array([9, 43, 43], dtype=np.uint8)
    yellow_hsv_max = np.array([34, 255, 255], dtype=np.uint8)
    img = cv2.imread('lands/land_cam_2024-01-05 17_36_33.jpg')
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    inrange_img = cv2.inRange(hsv_img, yellow_hsv_min, yellow_hsv_max)
    inrange_img = cv2.cvtColor(inrange_img, cv2.COLOR_GRAY2BGR)
    merge = np.hstack((img, inrange_img))
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    imgSplit()