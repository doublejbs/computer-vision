import myfunction
import cv2
import numpy as np
import time
import math


gKern = myfunction.get_gaussian_filter_2d(7, 1.5)

lenna = myfunction.cross_correlation_2d(cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE), gKern)
shape = myfunction.cross_correlation_2d(cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE), gKern)

def compute_image_gradient ( img , title, gKern):
    print('init!')
    #_img = myfunction.cross_correlation_2d(img, gKern)
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    gradientX = myfunction.cross_correlation_2d(img, sobelX)
    gradientY = myfunction.cross_correlation_2d(img, sobelY)

    magnitude = np.sqrt(gradientX*gradientX + gradientY*gradientY)
    magnitude = magnitude / magnitude.max() * 255
    direction = np.arctan2(gradientY, gradientX)
    return (magnitude, direction)

#cv2.waitKey(0)
def non_maximum_suppression_dir(mag, dir):
    h, w = mag.shape
    _mag = np.zeros((h, w))
    angle = dir*180/math.pi
    for y in range(1, h-1):
        for x in range(1, w-1):
            if mag[y][x] == 0:
                _mag[y][x] = 0
                continue
            if angle[y, x] < 0 : angle[y, x] += 180
            # 0도
            if (0 <= angle[y, x] < 22.5) or (157.5 <= angle[y, x] <= 180):
                mag1 = mag[y, x + 1]
                mag2 = mag[y, x - 1]
            # 45도
            elif (22.5 <= angle[y, x] < 67.5):
                mag1 = mag[y + 1, x - 1]
                mag2 = mag[y - 1, x + 1]
            # 90도
            elif (67.5 <= angle[y, x] < 112.5):
                mag1 = mag[y + 1, x]
                mag2 = mag[y - 1, x]
            # 135
            elif (112.5 <= angle[y, x] < 157.5):
                mag1 = mag[y - 1, x - 1]
                mag2 = mag[y + 1, x + 1]
            if (mag[y, x] >= mag1) and (mag[y, x] >= mag2):
                _mag[y, x] = mag[y, x]
            else:
                _mag[y, x] = 0
    return _mag

startTime = time.time()
lennamag, lennadir = compute_image_gradient(lenna, 'lenna', gKern)
print('Elapsed time of compute_image_gradient for lenna.png : ' + str((time.time()-startTime)) + ' s')
mask_gray = cv2.normalize(src=lennamag, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imshow('2 - 2 (d) lenna.png', mask_gray)
cv2.imwrite('./result/part_2_edge_raw_lenna.png', mask_gray)


startTime = time.time()
_lennamag = non_maximum_suppression_dir(lennamag, lennadir)
print('Elapsed time for nms on lenna.png : ' + str(time.time()-startTime) + 's')
mask_gray = cv2.normalize(src=_lennamag, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imwrite('./result/part_2_edge_sup_lenna.png', mask_gray)
cv2.imshow('2 - 3 (d) lenna.png', mask_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()





startTime = time.time()
shapemag, shapedir = compute_image_gradient(shape, 'shapes', gKern)
print('Elapsed time of compute_image_gradient for shapes.png : ' + str((time.time()-startTime)) + ' s')
mask_gray = cv2.normalize(src=shapemag, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imshow('2 - 2 (d) shapes.png', mask_gray)
cv2.imwrite('./result/part_2_edge_raw_shapes.png', mask_gray)
startTime = time.time()


_shapemag = non_maximum_suppression_dir(shapemag, shapedir)
print('Elapsed time for nms on shpaes.png : ' + str(time.time()-startTime) + 's')
mask_gray = cv2.normalize(src=_shapemag, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imshow('2 - 3 (d) shapes.png', mask_gray)
cv2.imwrite('./result/part_2_edge_sup_shapes.png', mask_gray)
cv2.waitKey(0)