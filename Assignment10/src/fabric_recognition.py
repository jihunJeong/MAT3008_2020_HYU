import math
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

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
    print("Creating Image matrix",end=" ... ")

    data = np.zeros((64, 64), dtype=np.float32)
    data = image[y:y+64,x:x+64]
	
    print("DONE")
    return data

def averageDft(image):
    avgdft = np.zeros((64, 64), dtype=np.float32)
    avgdft += np.fft.fft2(createImageMatrix(image, 0, 0))
    avgdft += np.fft.fft2(createImageMatrix(image, 0, 64))
    avgdft += np.fft.fft2(createImageMatrix(image, 64, 0))
    avgdft += np.fft.fft2(createImageMatrix(image, 64, 64))
    avgdft += np.fft.fft2(createImageMatrix(image))
    avgdft = avgdft/5


if __name__ == "__main__":
    imgs = readImages('../image')
    data_arr = createDataMatrix(imgs)
    for data in data_arr:
        show_magnitude(data)