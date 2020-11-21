import sys
import os
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import math

global g_r
global g_b
global r_b
global y_u
global y_v
global u_v

def readImages(folder):
    images = []
    idx = 1
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_COLOR)

        if img is not None:
            images.append(img)
        idx += 1

    return images

def show(b, g, r, z):
    img_r = cv2.merge([z, z, r])
    img_g = cv2.merge([z, g, z])
    img_b = cv2.merge([b, z, z])
    
    cv2.imshow("Red", img_r)
    cv2.imwrite('../images/'+"red_result"+'.png', img_r)		
    
    cv2.imshow("Green", img_g)
    cv2.imwrite('../images/'+"green_result"+'.png', img_g)		
    
    cv2.imshow("Blue", img_b)
    cv2.imwrite('../images/'+"blue_result"+'.png', img_b)		
    
    cv2.waitKey(0)

def compare_correlation(x, y):
    a1 = x.flatten()
    a2 = y.flatten()

    #print(np.corrcoef(a1, a2))
    return np.corrcoef(a1, a2)

def partitioned(img):
    global g_r, g_b, r_b, y_u, y_v, u_v
    b, g, r = cv2.split(img)
    z = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")

    show(b, g, r, z)
    g_r += compare_correlation(g, r)
    g_b += compare_correlation(g, b)
    r_b += compare_correlation(r, b)

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    #show(y, u, v, z)
    y_u += compare_correlation(y, u)
    y_v += compare_correlation(y, v)
    u_v += compare_correlation(u, v)


    
if __name__ == "__main__":
    global g_r, g_b, r_b, y_u, y_v, u_v
    g_r = np.zeros((2, 2), dtype="float64")
    g_b = np.zeros((2, 2), dtype="float64")
    r_b = np.zeros((2, 2), dtype="float64")
    y_u = np.zeros((2, 2), dtype="float64")
    y_v = np.zeros((2, 2), dtype="float64")
    u_v = np.zeros((2, 2), dtype="float64")

    imgs = readImages('../image')

    for img in imgs:
        partitioned(img)

    g_r = g_r / 10
    g_b = g_b / 10
    r_b = r_b / 10
    y_u = y_u / 10
    y_v = y_v / 10
    u_v = u_v / 10
    print(g_r)
    print(g_b)
    print(r_b)
    print(y_u)
    print(y_v)
    print(u_v)