import cv2
import numpy as np
import myfunction
import time

gKern = myfunction.get_gaussian_filter_2d(7, 1.5)
lenna = myfunction.cross_correlation_2d(cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE), gKern)
shape = myfunction.cross_correlation_2d(cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE), gKern)

def compute_corner_response(img, title):
    sobelX = np.array([[1, 0, -1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    h, w = img.shape

    gradientX = myfunction.cross_correlation_2d(img, sobelX)
    gradientY = myfunction.cross_correlation_2d(img, sobelY)

    wSize = 5
    Ix2 = np.square(gradientX)
    Iy2 = np.square(gradientY)
    Ixy = gradientY*gradientX
    R = np.zeros((h, w))
    for y in range(wSize//2, h-wSize//2-1):
        for x in range(wSize//2, w-wSize//2-1):
            sumxx = np.sum(Ix2[y-wSize//2:y+wSize//2+1, x-wSize//2:x+wSize//2+1])
            sumyy = np.sum(Iy2[y-wSize//2:y+wSize//2+1, x-wSize//2:x+wSize//2+1])
            sumxy = np.sum(Ixy[y-wSize//2:y+wSize//2+1, x-wSize//2:x+wSize//2+1])
            det = sumxx * sumyy - sumxy*sumxy
            trace = sumxx + sumyy
            r = det - 0.04*(trace*trace)
            if r<0: r=0
            R[y, x] = r
    # det = ad - bc = sumxx*sumyy - sumxy*sumxy  trace = a + d = sumxx + sumyy
    R = (R-np.amin(R))/(np.amax(R)-np.amin(R))
    mask_gray = cv2.normalize(src=R, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imwrite('./result/part_3_corner_ raw_'+title, mask_gray)
    cv2.imshow(title, mask_gray)
    return R

def non_maximum_suppression_win ( R , winSize ):
    h, w = R.shape
    _R = np.zeros((h, w))
    for y in range(winSize//2, h-winSize//2-1):
        for x in range(winSize//2, w-winSize//2-1):
            if R[y, x] >= 0.1 and R[y, x] >= np.amax(R[y-winSize//2:y+winSize//2+1, x-winSize//2:x+winSize//2+1]):
                _R[y, x] = R[y, x]
    return _R

startTime = time.time()
lennaR = compute_corner_response(lenna, 'lenna.png')
print('Elapsed time of compute_corner_response for lenna.png : ' + str(time.time()-startTime) + 's')

lh, lw = lenna.shape
lenna = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
coloredLenna = cv2.cvtColor(lenna, cv2.COLOR_GRAY2BGR)
for y in range(lh):
    for x in range(lw):
        if lennaR[y, x] > 0.1:
            coloredLenna[y, x][0] = 0
            coloredLenna[y, x][1] = 255
            coloredLenna[y, x][2] = 0
cv2.imshow('colorlenna', coloredLenna)
cv2.imwrite('./result/part_3_corner_bin_lenna.png', coloredLenna)

coloredLenna2 = cv2.cvtColor(lenna, cv2.COLOR_GRAY2BGR)
startTime = time.time()
_R = non_maximum_suppression_win(lennaR, 11)
print('Elapsed time of non_maximum_suppresiion_win on lenna.png : ' + str(time.time() - startTime) + 's')
_Rh, _Rw = _R.shape
for y in range(_Rh):
    for x in range(_Rw):
        if _R[y, x] > 0:
            coloredLenna2 = cv2.circle(coloredLenna2, (x, y), 4, (0, 255, 0), 2)
            #coloredLenna2[y, x] = (0, 255, 0)
cv2.imshow('lennenen', coloredLenna2)
cv2.imwrite('./result/part_3_corner_sup_lenna.png', coloredLenna2)
cv2.waitKey(0)
cv2.destroyAllWindows()


startTime = time.time()
shapeR = compute_corner_response(shape, 'shapes.png')
print('Elapsed time of compute_corner_response for shapes.png : ' + str(time.time()-startTime) + 's')

sh, sw = shape.shape
shape = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
coloredShape = cv2.cvtColor(shape, cv2.COLOR_GRAY2BGR)
for y in range(sh):
    for x in range(sw):
        if shapeR[y, x] > 0.1:
            coloredShape[y, x][0] = 0
            coloredShape[y, x][1] = 255
            coloredShape[y, x][2] = 0
cv2.imshow('colorshape', coloredShape)
cv2.imwrite('./result/part_3_corner_bin_shapes.png', coloredShape)

coloredShape2 = cv2.cvtColor(shape, cv2.COLOR_GRAY2BGR)
startTime = time.time()
_R = non_maximum_suppression_win(shapeR, 11)
print('Elapsed time of non_maximum_suppresiion_win on shapes.png : ' + str(time.time() - startTime) + 's')
_Rh, _Rw = _R.shape
for y in range(_Rh):
    for x in range(_Rw):
        if _R[y, x] > 0:
            cv2.circle(coloredShape2, (x, y), 4, (0, 255, 0), 2)
cv2.imshow('dsfsdds', coloredShape2)
cv2.imwrite('./result/part_3_corner_sup_shapes.png', coloredShape2)

cv2.waitKey(0)