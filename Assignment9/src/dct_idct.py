import sys
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import math
            
def rgb_show(r, g, b, z):
    img_r = cv2.merge([z, z, r])
    img_g = cv2.merge([z, g, z])
    img_b = cv2.merge([b, z, z])

    cv2.imshow("Red", img_r)
    cv2.imshow("Green", img_g)
    cv2.imshow("Blue", img_b)
    cv2.waitKey(0)

def partitioned(image):
    print("Partition Image",end=" ... ")

    arr = np.asarray(image)
    w = len(arr[0]) // 16
    h = len(arr) // 16
    arr = np.split(arr, h, 0)
    arr = np.array([np.split(x, w, 1) for x in arr])
    
    print("DONE")
    return arr

def dct(data):
    coeffi = []
    for v in range(16):
        for u in range(16):
            Fvu, cv, cu = 0.0, 1, 1

            if v == 0:
                cv = 1 / math.sqrt(2)
            if u == 0:
                cu = 1 / math.sqrt(2)

            for y in range(16):
                for x in range(16):
                    cosv = (v * math.pi * (2*y +1)) / (32)
                    cosu = (u * math.pi * (2*x +1)) / (32)
                    Fvu += data[y][x] * math.cos(cosv) * math.cos(cosu)
            Fvu = (Fvu * cv * cu) / 8
            coeffi.append([Fvu, v, u])
    coeffi = sorted(coeffi, key=lambda x : x[0], reverse=True)
    coeffi = coeffi[:16]
    
    return coeffi

def idct(data, coeffi):
    narr = [[0.0 for x in range(16)] for y in range(16)]
    for y in range(16):
        for x in range(16):
            Syx, cv, cu = 0.0, 1, 1

            for Fvu, v, u in coeffi:
                if v == 0:
                    cv = 1 / math.sqrt(2)
                else :
                    cv = 1
                if u == 0:
                    cu = 1 / math.sqrt(2)
                else :
                    cu = 1
                cosv = (v * math.pi * (2*y+1)) / (32)
                cosu = (u * math.pi * (2*x+1)) / (32)
                Syx += (cv * cu * Fvu) * math.cos(cosv) * math.cos(cosu)
            Syx = Syx / 4
            narr[y][x] = Syx 

    return narr               

def rebuildImage(img_arr):
    ans = [[]]
    for i in range(len(img_arr)):
        new = [[]]
        for j in range(len(img_arr[0])):
            cf = dct(img_arr[i][j])
            narr = idct(img_arr[i][j], cf)
            if j == 0:
                new = narr
                continue            
            new = np.concatenate((new, narr), axis=1)
        if i == 0:
            ans = new
            continue
        ans = np.concatenate((ans, new), axis=0)
    return ans.astype(np.uint8)

if __name__ == "__main__":
    img1 = cv2.imread("../images/1.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("../images/2.png", cv2.IMREAD_COLOR)
    img3 = cv2.imread("../images/3.jpg", cv2.IMREAD_COLOR)

    b1, g1, r1 = cv2.split(img1)
    z = np.zeros((img1.shape[0], img1.shape[1]), dtype="uint8")
    rgb_show(r1, g1, b1, z)

    b2, g2, r2 = cv2.split(img2)
    z = np.zeros((img2.shape[0], img2.shape[1]), dtype="uint8")
    rgb_show(r2, g2, b2, z)

    b3, g3, r3 = cv2.split(img3)
    z = np.zeros((img3.shape[0], img3.shape[1]), dtype="uint8")
    rgb_show(r3, g3, b3, z)
    
    rimg_arr = partitioned(r1)
    print("Dct, Idct Red...",end=" ")
    r = rebuildImage(rimg_arr)
    print(len(r))
    print(len(r[0]))
    print("Done")

    gimg_arr = partitioned(g1)
    print("Dct, Idct Green...",end=" ")
    g = rebuildImage(gimg_arr)
    print("Done")

    bimg_arr = partitioned(b1)
    print("Dct, Idct Blue...",end=" ")
    b = rebuildImage(bimg_arr)
    print("Done")       

    rgb_show(r, g, b, z)
    inverseImage = cv2.merge((r,g,b))
    cv2.imshow("Inverse", inverseImage)
    cv2.waitKey(0)