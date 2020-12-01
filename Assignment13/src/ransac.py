import math
import numpy as np
from matplotlib import pyplot as plt
import random

if __name__ == "__main__":
    xsample = np.zeros(12)
    ysample = np.zeros(12)
    for i in range(12):
        x = i - 5
        noise = np.random.normal(0,2)
        y = 2*x - 1 + noise
        xsample[i], ysample[i] = x, y
    
    A = np.vstack([xsample, np.ones(len(xsample))]).T
    m, c = np.linalg.lstsq(A, ysample, rcond=None)[0]
    err = np.linalg.lstsq(A, ysample, rcond=None)[1]
    print("{} {} {}".format(m, c, err))
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.scatter(xsample, ysample,c='g',label='Data')
    plt.plot(xsample, xsample*m+c,'b',label='LSQ')
    plt.plot(xsample, xsample*2-1,'r',label='Origin')

    cnt, min_m, min_c, min_e = 0, 1000, 1000, 1000
    min_sample = np.zeros(10)
    while True:
        if cnt >= 10000:
            break
        nx_li = sorted(random.sample(list(xsample), 10))
        ny_li = np.zeros(10)
        for i in range(len(nx_li)):
            ny_li[i] = ysample[int(nx_li[i]+5)]

        nx_li = np.array(nx_li)
        A = np.vstack([nx_li, np.ones(len(nx_li))]).T
        m, c = np.linalg.lstsq(A, ny_li, rcond=None)[0]
        err = np.linalg.lstsq(A, ny_li, rcond=None)[1]
        if min_e > err:
            min_m, min_c, min_e = m, c, err
            min_sample = nx_li.copy()
            #print(min_sample,end=" ")
            #print("{} {}".format(min_e, cnt))
        cnt += 1

    print("{} {} {}".format(min_m, min_c, min_e))
    plt.plot(xsample, xsample*min_m+min_c,label='Ransac')

    plt.legend()
    plt.show()