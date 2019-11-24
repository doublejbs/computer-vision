import cv2
import numpy as np
import math
import time

def setPadding(img, wSize):
    h, w = img.shape
    for stack in range(wSize//2):
        img = np.vstack((img[0, : ], img))
        img = np.vstack((img, img[h-1, : ]))
        img = np.hstack((img[ : , 0][:, None], img))
        img = np.hstack((img, img[ : , w-1][:, None]))
    return img

def setPaddingHoriz(img, wSize):
    h, w = img.shape
    for stack in range(wSize // 2):
        img = np.hstack((img[:, 0][:, None], img))
        img = np.hstack((img, img[:, w - 1][:, None]))
    return img

def setPaddingVert(img, wSize):
    h, w = img.shape
    for stack in range(wSize // 2):
        img = np.vstack((img[0, :], img))
        img = np.vstack((img, img[h - 1, :]))
    return img

def cross_correlation_1d(img, kernel):
    h, w = img.shape
    res = np.zeros((h, w))
    isHorizontal = False
    wSize = kernel.shape[0]
    if len(kernel.shape) == 1: isHorizontal = True
    if isHorizontal:
        paddedImg = setPaddingHoriz(img, wSize)
        for y in range(h):
            for x in range(w):
                res[y, x] = np.sum(paddedImg[y, x:x + wSize] * kernel)
    else:
        paddedImg = setPaddingVert(img, wSize)
        for y in range(h):
            for x in range(w):
                res[y, x] = np.sum(paddedImg[y:y + wSize, x] * kernel.reshape(kernel.shape[0], ))
    return res
def cross_correlation_2d(img, kernel):
    h, w = img.shape
    wSize = kernel.shape[0]
    paddedImg = setPadding(img, wSize)
    res = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            res[y, x] = np.sum(paddedImg[y:y+wSize, x:x+wSize] * kernel)
    return res

def get_gaussian_filter_1d ( size , sigma ):
    gKern = np.zeros((size, ))
    i = 0
    for x in range(-1*(size//2), size//2+1):
        gKern[i] = pow(math.e, -1*(pow(x, 2)/(2*pow(sigma, 2)))) / (sigma*math.sqrt(2*math.pi))
        i += 1
    return gKern

def get_gaussian_filter_2d(size, sigma):
    gKern = np.zeros((size, size))
    j = 0
    for y in range(-1*(size//2), size//2+1):
        i = 0
        for x in range(-1*(size//2), size//2+1):
            gKern[j, i] = (pow(math.e, -1*((pow(x, 2)+pow(y, 2))/(2*pow(sigma, 2)))) / (2*math.pi*pow(sigma, 2)))
            i += 1
        j += 1
    return gKern

def separate_gaussian_kernel(kernel):
    horizontal = kernel[0, :]
    return horizontal, horizontal.reshape(kernel.shape[0], 1)


print('get_gaussian_filter_1d(5,1) : ', get_gaussian_filter_1d(5,1))
print('get_gaussian_filter_2d(5,1) : ', get_gaussian_filter_2d(5,1))

#lenna (d)
imgList = []
print('Performing gaussian filter on lenna.png...')
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE )
for j in range(5, 18, 6):
    tList = []
    for i in range(1, 12, 5):
        gKern = get_gaussian_filter_2d(j, i)
        _2dgauss = cross_correlation_2d(img, gKern)
        print('done (' + str(j) + ', ' + str(i) + ')')
        mask_gray = (_2dgauss - np.amin(_2dgauss)) / (np.amax(_2dgauss) - np.amin(_2dgauss))
        point = 10, 40
        cv2.putText(mask_gray, str(j) + ' X ' + str(j) + ' S = ' + str(i), point,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        _mask_gray = cv2.resize(mask_gray, dsize=(0, 0), fx=0.3, fy=0.3)
        tList.append(_mask_gray)
    imgList.append(tList)
temp1 = np.hstack((imgList[0][0], imgList[0][1]))
temp1 = np.hstack((temp1, imgList[0][2]))
temp2 = np.hstack((imgList[1][0], imgList[1][1]))
temp2 = np.hstack((temp2, imgList[1][2]))
temp3 = np.hstack((imgList[2][0], imgList[2][1]))
temp3 = np.hstack((temp3, imgList[2][2]))
v = np.vstack((temp1, temp2))
v = np.vstack((v, temp3))

cv2.imshow('1 - (d) lenna.png', v)
print('done gaussian filter on lenna.png!!')
cv2.imwrite('./result/part_1_result_gaussian_filtered_lenna.png', v*255)

#lenna (e)
gKern = get_gaussian_filter_2d(5, 11)
hKern, vKern = separate_gaussian_kernel(gKern)
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE )

startTime = time.time()
res2 = cross_correlation_2d(img, gKern)
print('//gaussian filter (5, 11)//')
print("Elapsed time of 2d gaussian on lenna.png : ", (time.time()-startTime), "s")
startTime = time.time()
res = cross_correlation_1d(img, hKern)
_res = cross_correlation_1d(res, vKern)
print("Elapsed time of 1d gaussian sequentially on lenna.png : ", (time.time()-startTime), "s")

diffSum = 0
print('difference map')
dif = np.zeros((res2.shape[0], res2.shape[1]))
res2 = cv2.normalize(src=res2, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
_res = cv2.normalize(src=_res, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
for y in range(len(_res)):
    for x in range(len(_res[0])):
        dif[y, x] = abs(res2[y, x] - _res[y, x])
        #print(dif[y, x])
        diffSum += abs(res2[y, x] - _res[y, x])
print('sum of difference between 1d and 2d gaussian on lenna.png :', diffSum)
#dif = cv2.normalize(src=dif, dst=None, alpha=0, beta=128, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#dif = (dif - np.amin(dif)) / (np.amax(dif) - np.amin(dif)) * 255
cv2.imshow('difference map of lenna.png', dif)
cv2.waitKey(0)
cv2.destroyAllWindows()

#shapes (d)
imgList = []
img = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE )
print('Performing gaussian filter on shapes.png....')
for j in range(5, 18, 6):
    tList = []
    for i in range(1, 12, 5):
        gKern = get_gaussian_filter_2d(j, i)
        _2dgauss = cross_correlation_2d(img, gKern)
        print('done (' + str(j) + ', ' + str(i) + ')')
        mask_gray = (_2dgauss - np.amin(_2dgauss)) / (np.amax(_2dgauss) - np.amin(_2dgauss))
        point = 10, 40
        cv2.putText(mask_gray, str(j) + ' X ' + str(j) + ' S = ' + str(i), point,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        _mask_gray = cv2.resize(mask_gray, dsize=(0, 0), fx=0.3, fy=0.3)
        tList.append(_mask_gray)
    imgList.append(tList)
temp1 = np.hstack((imgList[0][0], imgList[0][1]))
temp1 = np.hstack((temp1, imgList[0][2]))
temp2 = np.hstack((imgList[1][0], imgList[1][1]))
temp2 = np.hstack((temp2, imgList[1][2]))
temp3 = np.hstack((imgList[2][0], imgList[2][1]))
temp3 = np.hstack((temp3, imgList[2][2]))
v = np.vstack((temp1, temp2))
v = np.vstack((v, temp3))
cv2.imshow('1 - (d) shapes.png', v)
print('done gaussian filter on shapes.png!!')
cv2.imwrite('./result/part_1_result_gaussian_filtered_shapes.png', v*255)


#shape (e)
gKern = get_gaussian_filter_2d(5, 11)
hKern, vKern = separate_gaussian_kernel(gKern)
img = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE )

startTime = time.time()
res2 = cross_correlation_2d(img, gKern)
print("Elapsed time of 2d gaussian on shapes.png : ", (time.time()-startTime), "s")
startTime = time.time()
res = cross_correlation_1d(img, hKern)
_res = cross_correlation_1d(res, vKern)
print("Elapsed time of 1d gaussian sequentially on shapes.png : ", (time.time()-startTime), "s")

diffSum = 0
dif = np.zeros((res2.shape[0], res2.shape[1]))
res2 = cv2.normalize(src=res2, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
_res = cv2.normalize(src=_res, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
for y in range(len(_res)):
    for x in range(len(_res[0])):
        dif[y, x] = abs(res2[y, x] - _res[y, x])
        #print(dif[y, x])
        diffSum += abs(res2[y][x] - _res[y][x])
print('sum of difference between 1d and 2d gaussian on shapes.png :', diffSum)
#dif = cv2.normalize(src=dif, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imshow('difference map of shapes.png', dif)
cv2.waitKey(0)