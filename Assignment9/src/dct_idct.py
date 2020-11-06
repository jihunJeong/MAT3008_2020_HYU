import sys
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import math

def partitioned(image):
    print("Partition Image",end=" ... ")

    arr = np.asarray(image)
    arr = np.split(arr, 16, 0)
    arr = np.array([np.split(x, 16, 1) for x in arr])
    
    print("DONE")
    return arr

def dct(data):
    print("Dct ...",end=" ")
    
    coeffi = []
    for v in range(16):
        for u in range(16):
            Fvu, cv, cu = 1.0, 1, 1

            if v == 0:
                cv = 1 / math.sqrt(2)
            if u == 0:
                cu = 1 / math.sqrt(2)

            for y in range(16):
                for x in range(16):
                    cosv = (v * math.pi * (2*y +1)) / (32)
                    sinv = (u * math.pi * (2*x +1)) / (32)
                    Fvu += data[y][x] * math.cos(cosv) * math.sin(sinv)
            Fvu = (Fvu * cv * cu) / 8
            coeffi.append([Fvu, v, u])
    coeffi = sorted(coeffi, key=lambda x : x[0], reverse=True)
    coeffi = coeffi[:16]
    print("Done")
    return coeffi
            
def rgb_show(r, g, b, z):
    img_r = cv2.merge([z, z, r])
    img_g = cv2.merge([z, g, z])
    img_b = cv2.merge([b, z, z])

    cv2.imshow("Red", img_r)
    cv2.imshow("Green", img_g)
    cv2.imshow("Blue", img_b)
    cv2.waitKey(0)

if __name__ == "__main__":
    img1 = cv2.imread("../images/1.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("../images/2.png", cv2.IMREAD_COLOR)
    img3 = cv2.imread("../images/3.jpg", cv2.IMREAD_COLOR)

    b1, g1, r1 = cv2.split(img1)
    z = np.zeros((img1.shape[0], img1.shape[1]), dtype="uint8")
    rgb_show(r1, g1, b1, z)
    
    img_arr = partitioned(r1)
    for i in range(len(img_arr)):
        for j in range(len(img_arr[0])):
            cf = dct(img_arr[i][j])

    b2, g2, r2 = cv2.split(img2)
    z = np.zeros((img2.shape[0], img2.shape[1]), dtype="uint8")
    rgb_show(r2, g2, b2, z)

    b3, g3, r3 = cv2.split(img3)
    z = np.zeros((img3.shape[0], img3.shape[1]), dtype="uint8")
    rgb_show(r3, g3, b3, z)
    
    