import math
import random
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity

def readImages(folder):
    images = []
    idx = 1
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            images.append(img)
        idx += 1

    return images

def createDataMatrix(images, y, x):
    print("Creating data matrix",end=" ... ")

    numImages = len(images)
    data = np.zeros((numImages, 64, 64), dtype=np.float32)
    for i in range(0, numImages):
        image = images[i]
        data[i,:] = image[y:y+64,x:x+64]
	
    print("DONE")
    return data

def show_magnitude(data, idx=0):
    dft = np.fft.fft2(data)
    scaled = minmax_scale(abs(dft), feature_range=(0,255))
    scaled = np.fft.fftshift(scaled)
    scaled = scaled.astype(np.uint8)
    output = cv2.resize(scaled, dsize=(128, 128))
    cv2.imshow("result", output)
    idft = np.fft.ifft2(dft)
    idft = idft.astype(np.uint8)
    iout = cv2.resize(idft, dsize=(128, 128))
    cv2.imshow("new", iout)
    cv2.waitKey()
    if idx != 0:
        cv2.imwrite('../result/dft'+str(idx)+'.png', output)
        cv2.imwrite('../result/idft'+str(idx)+'.png', iout)
    
def createImageMatrix(image, y=32, x=32):
    data = np.zeros((64, 64), dtype=np.float32)
    data = image[y:y+64,x:x+64]
	
    return data

global boundary

def averageDft(image):
    avgdft = np.zeros((64, 64), dtype=np.complex128)
    data = np.zeros((5, 99), dtype=np.float32)
    d = np.zeros((64, 64), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            d = np.fft.fft2(createImageMatrix(image,64*i,64*j))
            avgdft += d
            for k in range(10):
                for m in range(10):
                    if k == 0 and m == 0:
                        continue
                    data[i*2+j][k*10+m-1] = abs(d[k][m])
    
    d = np.fft.fft2(createImageMatrix(image,32,32))
    for i in range(10):
        for j in range(10):
            if i == 0 and j == 0:
                continue
            data[4][i*10+j-1] = abs(d[i][j])

    sample = np.zeros((1,99), dtype=np.float32)
    avgdft += d
    avgdft = avgdft/5
    for i in range(10):
        for j in range(10):
            if i == 0 and j == 0:
                continue
            sample[0][i*10+j-1] = abs(avgdft[i][j])

    global boundary
    val = cosine_similarity(sample, data)
    for i in range(5):
        idx = int(math.floor(val[0][i] * 10))
        boundary[idx] += 1

    return sample

if __name__ == "__main__":
    imgs = readImages('../image')
    data_arr = createDataMatrix(imgs, 32, 32)
    
    for i in range(len(data_arr)):
        show_magnitude(data_arr[i], i+1)
    
    global boundary
    boundary = np.zeros(10, dtype=np.int32)
    
    sample_set = np.zeros((20,99), dtype=np.float32)
    for i in range(len(imgs)):
        sample_set[i] = averageDft(imgs[i])
    
    for i in range(10):
        print("{}~{} : {}".format(i/10, (i+1)/10, boundary[i]))
    print("")

    boundary = np.zeros(10, dtype=np.int32)
    for i in range(20):
        for j in range(i, 20):
            if i == j:
                continue
            val = cosine_similarity([sample_set[i]],[sample_set[j]])
            idx = int(math.floor(val[0][0] * 10))
            boundary[idx] += 1
    for i in range(10):
        print("{}~{} : {}".format(i/10, (i+1)/10, boundary[i]))
    print("")
    
    for bd in range(10):
        crt, wrg, tcnt = 0, 0, 0
        thres = 0.7 + (bd/100)
        for testtotal in range(20):
            for idx in range(len(imgs)):
                for _ in range(5):
                    tcnt += 1
                    yrand = random.randint(0,64)
                    xrand = random.randint(0,64)
                    d = np.fft.fft2(createImageMatrix(imgs[idx],yrand,xrand))
                    temp = np.zeros(99, dtype=np.float32)
                    for i in range(10):
                        for j in range(10):
                            if i == 0 and j == 0:
                                continue
                            temp[i*10+j-1] = abs(d[i][j])
                    
                    semi_flag, flag = False, True
                    for i in range(20):
                        val = cosine_similarity([sample_set[i]], [temp])
                        if val >= thres:
                            if i == idx:
                                semi_flag = True
                            else :
                                flag = False
                        if i == 19 and semi_flag == False:
                            flag = False
                    if flag:
                        crt += 1
                    else :
                        wrg += 1
        print("Result {:.2f} - Test : {}, Correct : {}, Wrong : {}, Rate : {:.2f}%".format(thres, tcnt, crt, wrg, (crt/tcnt)*100))
    
    crt, wrg, tcnt = 0, 0, 0
    for idx in range(len(imgs)):
        for _ in range(5):
            tcnt += 1
            yrand = random.randint(0,64)
            xrand = random.randint(0,64)
            d = np.fft.fft2(createImageMatrix(imgs[idx],yrand,xrand))
            temp = np.zeros(99, dtype=np.float32)
            for i in range(10):
                for j in range(10):
                    if i == 0 and j == 0:
                        continue
                    temp[i*10+j-1] = abs(d[i][j])
                    
            semi_flag, flag = False, True
            for i in range(20):
                val = cosine_similarity([sample_set[i]], [temp])
                if val >= 0.75:
                    if i == idx:
                        semi_flag = True
                    else :
                        flag = False
                if i == 19 and semi_flag == False:
                    flag = False
            if flag:
                crt += 1
            else :
                wrg += 1
    print("Test : {}, Correct : {}, Wrong : {}, Rate : {:.2f}%".format(tcnt, crt, wrg, (crt/tcnt)*100))