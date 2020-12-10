from sklearn.cluster import MeanShift, KMeans
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def readImages(folder):
    images = []
    idx = 1
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_COLOR)
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        rst_lab = lab.reshape((len(img)*len(img[0]),3))
        lab = rst_lab.reshape((96, 128, 3))
        lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imshow("Result", lab)
        cv2.waitKey(0)

        if img is not None:
            images.append(rst_lab)
        idx += 1

    return images

def mean_shift(img):
    rebuild = np.zeros((96*128, 3), dtype=np.float32)
    rst_img = np.zeros((96, 128, 3), dtype=np.float32)
    
    print(img.shape)
    clustering = MeanShift(bandwidth=12.5).fit(img)
    centers = clustering.cluster_centers_
    k = len(centers)
    label = clustering.labels_
    print("K is {}".format(len(centers)))
    for i in range(len(label)):
        rebuild[i] = centers[label[i]]
    
    rst_img = rebuild.reshape((96, 128, 3))
    lab = cv2.cvtColor(rst_img.astype(np.uint8), cv2.COLOR_LAB2BGR)
    lab = cv2.resize(lab, (0,0), fx=2, fy=2)
    cv2.imshow("Result", lab)
    cv2.waitKey(0)

    cv2.imwrite('../result/'+'mean_shift2'+'.png', lab)
        
    return k

def k_means(img, k):
    rebuild = np.zeros((96*128, 3), dtype=np.float32)
    rst_img = np.zeros((96, 128, 3), dtype=np.float32)
    
    print(img.shape)
    clustering = KMeans(n_clusters=k).fit(img)
    centers = clustering.cluster_centers_
    k = len(centers)
    label = clustering.labels_
    for i in range(len(label)):
        rebuild[i] = centers[label[i]]
    
    rst_img = rebuild.reshape((96, 128, 3))
    lab = cv2.cvtColor(rst_img.astype(np.uint8), cv2.COLOR_LAB2BGR)
    lab = cv2.resize(lab, (0,0), fx=2, fy=2)
    cv2.imshow("Result", lab)
    cv2.waitKey(0)

    cv2.imwrite('../result/'+'k_means2'+'.png', lab)
        
    return k

if __name__ == "__main__":
    imgs = readImages('../image/test')
    
    k = mean_shift(imgs[1])
    k_means(imgs[1], k)