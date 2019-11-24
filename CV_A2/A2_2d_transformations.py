import cv2
import numpy as np
import math

def get_transformed_image(img, M):
    plane = np.ones((801, 801, 3), np.uint8)*255
    iH, iW = img.shape
    originM = np.array([[1, 0, 400], [0, 1, 400], [0, 0, 1]])
    imgNp = np.ones((iH, iW, 3), np.int16)
    y = 0
    for j in range(-1*(iH//2), iH//2+1):
        x = 0
        for i in range(-1*(iW//2), iW//2+1):
            imgNp[y, x] = [j, i, 1]
            x += 1
        y += 1
    _mat = np.matmul(originM, M)
    for j in range(iH):
        for i in range(iW):
            imgNp[j, i] = np.matmul(_mat, imgNp[j, i])
            plane[imgNp[j, i][0], imgNp[j, i][1]] = img[j, i]
    cv2.arrowedLine(plane, (0, 400), (800, 400), (0, 0, 0), thickness=3, tipLength=0.05)
    cv2.arrowedLine(plane, (400, 800), (400, 0), (0, 0, 0), thickness=3, tipLength=0.05)
    return plane

img = cv2.imread('smile.png', cv2.IMREAD_GRAYSCALE)
#print(img)
M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
deg = 5
beforeF = False
retransDic = dict()
retransDic['a'] = 0
retransDic['d'] = 0
retransDic['w'] = 0
retransDic['s'] = 0
transDic = dict()
transDic['a'] = 0
transDic['d'] = 0
transDic['w'] = 0
transDic['s'] = 0

def transition(key):
    m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if key=='a':
        m = np.matmul(np.array([[1, 0, 0], [0, 1, -5], [0, 0, 1]]), m)
    elif key=='d':
        m = np.matmul(np.array([[1, 0, 0], [0, 1, 5], [0, 0, 1]]), m)
    elif key=='w':
        m = np.matmul(np.array([[1, 0, -5], [0, 1, 0], [0, 0, 1]]), m)
    elif key=='s':
        m = np.matmul(np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]]), m)
    return m

while True:
    k = cv2.waitKey(0)
    if k==ord('a'):
        m = np.matmul(np.array([[1, 0, 0], [0, 1, -5], [0, 0, 1]]), m)
        retransDic['d'] += 1
        transDic['a'] += 1
    elif k==ord('d'):
        m = np.matmul(np.array([[1, 0, 0], [0, 1, 5], [0, 0, 1]]), m)
        retransDic['a'] += 1
        transDic['d'] += 1
    elif k==ord('w'):
        m = np.matmul(np.array([[1, 0, -5], [0, 1, 0], [0, 0, 1]]), m)
        retransDic['s'] += 1
        transDic['w'] += 1
    elif k==ord('s'):
        m = np.matmul(np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]]), m)
        retransDic['w'] += 1
        transDic['s'] += 1
    elif k==ord('r'):
        m = np.matmul(np.array([[math.cos(math.radians(deg)), -1*math.sin(math.radians(deg)), 0], [math.sin(math.radians(deg)), math.cos(math.radians(deg)), 0], [0, 0, 1]]), m)
    elif k==ord('R'):
        m = np.matmul(np.array([[math.cos(math.radians(-1*deg)), -1*math.sin(math.radians(-1*deg)), 0], [math.sin(math.radians(-1*deg)), math.cos(math.radians(-1*deg)), 0], [0, 0, 1]]), m)
    elif k==ord('f'):
        m = np.matmul(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]), m)
        deg *= -1
    elif k==ord('F'):
        m = np.matmul(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), m)
    elif k==ord('x'):
        m = np.matmul(np.array([[1, 0, 0], [0, 0.95, 0], [0, 0, 1]]), m)
    elif k==ord('X'):
        m = np.matmul(m, np.array([[1, 0, 0], [0, 1.05, 0], [0, 0, 1]]))
    elif k==ord('y'):
        m = np.matmul(np.array([[0.95, 0, 0], [0, 1, 0], [0, 0, 1]]), m)
    elif k==ord('Y'):
        m = np.matmul(m, np.array([[1.05, 0, 0], [0, 1, 0], [0, 0, 1]]))
    elif k==ord('h'):
        m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif k==ord('q'):
        break
    res = get_transformed_image(img, m)
    cv2.imshow('res', res)




