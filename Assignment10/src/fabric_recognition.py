import math
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

def createDataMatrix(images, y=32, x=32):
    print("Creating data matrix",end=" ... ")

    numImages = len(images)
    data = np.zeros((numImages, 64, 64), dtype=np.float32)
    for i in range(0, numImages):
        image = images[i]
        data[i,:] = image[y:y+64,x:x+64]
	
    print("DONE")
    return data

def show_magnitude(data):
    dft = np.fft.fft2(data)
    scaled = minmax_scale(abs(dft), feature_range=(0,255))
    scaled = np.fft.fftshift(scaled)
    scaled = scaled.astype(np.uint8)
    output = cv2.resize(scaled, (0, 0), fx=10, fy=10)
    cv2.imshow("result", output)
    idft = np.fft.ifft2(dft)
    idft = idft.astype(np.uint8)
    iout = cv2.resize(idft, (0, 0), fx=10, fy=10)
    cv2.imshow("new", iout)
    cv2.waitKey()

    '''
    rec_img = recreate.reshape(sz)
	rec_img = rec_img.astype(np.uint8)
	output = cv2.resize(rec_img, (0, 0), fx=10, fy=10)
	cv2.imshow("Result", output)
	cv2.waitKey()
	cv2.imwrite('./eigenfaces/'+str(idx)+'.png', output)
	idx += 1
    '''

def createImageMatrix(image, y=32, x=32):
    data = np.zeros((64, 64), dtype=np.float32)
    data = image[y:y+64,x:x+64]
	
    return data

def averageDft(image):
    avgdft = np.zeros((64, 64), dtype=np.complex128)
    data = np.zeros((5, 99), dtype=np.complex128)
    d = np.zeros((64, 64), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            d = np.fft.fft2(createImageMatrix(image,64*i,64*j))
            avgdft += d
            for k in range(10):
                for m in range(10):
                    if k == 0 and m == 0:
                        continue
                    data[i*2+j][k*10+j-1] = d[k][m]

    d = np.fft.fft2(createImageMatrix(image))
    avgdft = (avgdft+d)/5
    for i in range(10):
        for j in range(10):
            if i == 0 and j == 0:
                continue
            data[4][i*10+j-1] = d[i][j]

    sample = np.zeros((1,99), dtype=np.complex128)
    for i in range(10):
        for j in range(10):
            if i == 0 and j == 0:
                continue
            sample[0][i*10+j-1] = avgdft[i][j]
    
    #print(cosine_similarity(abs(sample), abs(data)))
    for i in range(5):
        print(np.linalg.norm(abs(sample[0]) - abs(data[i])))



if __name__ == "__main__":
    imgs = readImages('../image')
    data_arr = createDataMatrix(imgs)
    #for data in data_arr:
        #show_magnitude(data)
    for i in range(len(imgs)):
        averageDft(imgs[i])