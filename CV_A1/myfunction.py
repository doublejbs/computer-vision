import numpy as np
import math
import cv2

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

def compute_image_gradient ( img , title, gKern):
    print('init!')
    _img = cross_correlation_2d(img, gKern)
    sobelX = np.array([[1, 0, -1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    gradientX = cross_correlation_2d(_img, sobelX)
    gradientY = cross_correlation_2d(_img, sobelY)

    magnitude = np.sqrt(np.add(np.square(gradientX), np.square(gradientY)))
    magnitude = magnitude / magnitude.max() * 255
    print('working...')
    # mask_gray = cv2.normalize(src=magnitude, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # cv2.imshow(title, mask_gray)
    # cv2.imwrite('./result/part_2_edge_raw_'+title+'.png', mask_gray)
    direction = np.arctan2(gradientY, gradientX)
    #print(direction.shape)
    #print(direction[0])
    return (magnitude, direction)
