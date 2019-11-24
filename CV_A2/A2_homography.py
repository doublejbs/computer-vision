import cv2
import numpy as np
import random
import time
import math

img1 = cv2.imread('cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('cv_desk.png', cv2.IMREAD_GRAYSCALE)
hp = cv2.imread('hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
h1, w1 = img1.shape
h2, w2 = img2.shape

orb = cv2.ORB_create()

kp1 = orb.detect(img1, None)
kp1, des1 = orb.compute(img1, kp1)
kp2 = orb.detect(img2, None)
kp2, des2 = orb.compute(img2, kp2)
kp3 = orb.detect(hp, None)
kp3, des3 = orb.compute(hp, kp3)

d1H, d1W = des1.shape
d2H, d2W = des2.shape
d3H, d3W = des3.shape
ans1 = []
for j1 in range(d1H):
    minCount = 100000
    ansIdx = 0
    for j2 in range(d2H):
        count = 0
        for i1 in range(d1W):
            count += bin(des1[j1][i1]^des2[j2][i1]).count('1')
        if count < minCount:
            ansIdx = j2
            minCount = count
    ans1.append((j1, ansIdx, minCount))
ans1 = sorted(ans1, key=lambda x:x[2])
matches = []
# bf = cv2.BFMatcher()
# matches = bf.match(des1, des2)
# matches = sorted(matches, key = lambda x:x.distance)
#
for item in ans1:
    matches.append(cv2.DMatch(item[0], item[1], item[2]))
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
cv2.imshow('dd', img3)

srcP = []
destP = []
# for item in matches:
#     srcP.append([kp1[item.queryIdx].pt[0], kp1[item.queryIdx].pt[1]])
#     destP.append([kp2[item.trainIdx].pt[0], kp2[item.trainIdx].pt[1]])
for item in matches:
    srcP.append(kp1[item.queryIdx].pt)
    destP.append(kp2[item.trainIdx].pt)
srcP = np.array(srcP)
destP = np.array(destP)

def compute_homography ( srcP , destP ):
    #mean substraction
    normalizedSrcP = np.copy(srcP)
    normalizedDestP = np.copy(destP)
    w1 = np.sum(srcP[:, 0])/len(srcP)
    h1 = np.sum(srcP[:, 1])/len(srcP)
    w2 = np.sum(destP[:, 0])/len(destP)
    h2 = np.sum(destP[:, 1])/len(destP)
    trans1 = np.array([[1, 0, -1*w1], [0, -1, h1], [0, 0, 1]])
    trans2 = np.array([[1, 0, -1*w2], [0, -1, h2], [0, 0, 1]])
    maxSrc = 0
    maxDest = 0
    _normalizedSrcP = []
    _normalizedDestP = []
    for item in normalizedSrcP:
        item = np.array([item[0], item[1], 1])
        _item = np.matmul(trans1, item)
        #print('_ii', _item)
        _normalizedSrcP.append(_item)
        maxSrc = max(math.sqrt(_item[0]**2 + _item[1]**2), maxSrc)
    for item in normalizedDestP:
        item = np.array([item[0], item[1], 1])
        _item = np.matmul(trans2, item)
        _normalizedDestP.append(_item)
        maxDest = max(math.sqrt(_item[0]**2 + _item[1]**2), maxDest)
    _normalizedSrcP = np.array(_normalizedSrcP)
    _normalizedDestP = np.array(_normalizedDestP)
    #print('transition', _normalizedSrcP)


    scale1 = np.array([[1/maxSrc, 0, 0], [0, 1/maxSrc, 0], [0, 0, 1]])
    scale2 = np.array([[1/maxDest, 0, 0], [0, 1/maxDest, 0], [0, 0, 1]])
    __normalizedSrcP = []
    __normalizedDestP = []
    for item in _normalizedSrcP:
        item = np.array([item[0], item[1], 1])
        _item = np.matmul(scale1, item)
        __normalizedSrcP.append(_item)
    for item in _normalizedDestP:
        item = np.array([item[0], item[1], 1])
        _item = np.matmul(scale2, item)
        __normalizedDestP.append(_item)
    __normalizedSrcP = np.array(__normalizedSrcP)
    __normalizedDestP = np.array(__normalizedDestP)
    # print('scaledsrc', __normalizedSrcP)
    # print('sclaeddest', __normalizedDestP)

    t1 = np.matmul(scale1, trans1)
    t2 = np.matmul(scale2, trans2)
    # for item in normalizedSrcP:
    #     item[0] = item[0] - w1//2
    #     item[1] = h1//2 - item[1]
    #     maxSrc = max(max(abs(item[0]), abs(item[1])), maxSrc)
    # for item in normalizedDestP:
    #     item[0] = item[0] - w2//2
    #     item[1] = h2//2 - item[1]
    #     maxDest = max(max(abs(item[0]), abs(item[1])), maxDest)
    # print('origint', normalizedDestP)
    # print('__', __normalizedDestP)
    # for item in __normalizedDestP:
    #     print('changed', np.matmul(np.linalg.inv(t2), np.array([item[0], item[1], 1])))
    #print('changed', np.matmul(np.linalg.inv(t1), __normalizedSrcP))
    A = []
    # print('src', __normalizedSrcP)
    # print('des', __normalizedDestP)
    for j in range(len(_normalizedSrcP)):
        x = __normalizedSrcP[j][0]
        y = __normalizedSrcP[j][1]
        _x = __normalizedDestP[j][0]
        _y = __normalizedDestP[j][1]
        A.append([-1*x, -1*y, -1, 0, 0, 0, x*_x, y*_x, _x])
        A.append([0, 0, 0, -1*x, -1*y, -1, x*_y, y*_y, _y])
    A = np.array(A)
    #print(A)
    u, s, v = np.linalg.svd(A)
    h = v[8].reshape(3, 3)
    h /= h[2, 2]
    H = np.matmul(np.linalg.inv(t2), np.matmul(h, t1))
    #H /= H[2, 2]
    # for j in range(len(__normalizedSrcP)):
    #     print('calc', np.matmul(h, __normalizedSrcP[j]))
    #     print('ans', __normalizedDestP[j])
    # print('shape', H.shape)
    #H /= H[2, 2]
    return H

def compute_homography_ransac(srcP, destP, th):
    print('processing...')
    before = time.time()
    inlineSrcIdx = []
    inlineDestIdx = []
    done = []
    while (time.time()-before) <= 3:
        idx = []
        _srcP = []
        _destP = []
        for i in range(4):
            _go = True
            while _go and (time.time()-before) <= 3:
                #print('re')
                #print('i', idx)
                idx.append(random.randint(0, len(srcP)-1))
                if i==0: break
                if idx[i] not in idx[:i] and idx[i] not in inlineSrcIdx:
                    _go=False
                    break
                else:
                    if len(done) > len(srcP):
                        break
                    idx.pop(i)
        #print('idx', idx)
        for i in idx:
            _srcP.append(srcP[i])
            _destP.append(destP[i])
        _srcP = np.array(_srcP)
        _destP = np.array(_destP)
        h = compute_homography(_srcP, _destP)
        #h, s = cv2.findHomography(_srcP, _destP, cv2.RANSAC, 5.0)
        #count = 0
        for j in range(len(_srcP)):
            count = 0
            p = np.array([_srcP[j][0], _srcP[j][1], 1])
            _p = np.matmul(h, p)
            dif = np.array([_destP[j][0], _destP[j][1], 1]) - _p
            # print('p', p)
            # print('_p', _p)
            # print('ans', np.array([_destP[j][0], _destP[j][1], 1]))
            # print('dif', dif)
            _dif = math.sqrt(np.sum(dif**2))
            if _dif < th:
                inlineSrcIdx.append(idx[j])
    inlineSrc = []
    inlineDest = []
    for item in inlineSrcIdx:
        inlineSrc.append(srcP[item])
        inlineDest.append(destP[item])
    inlineSrc = np.array(inlineSrc)
    inlineDest = np.array(inlineDest)
    print(len(inlineSrc))
    print(inlineSrcIdx)
    h = compute_homography(inlineSrc, inlineDest)
    return h
idxes = [15, 4, 1, 25, 26, 12, 7, 29, 9, 11, 10, 6, 10, 14, 16, 13, 25, 29, 5, 22, 20, 2, 8, 25, 25, 17]
_srcP = []
_destP = []
for i in idxes:
    _srcP.append(srcP[i])
    _destP.append(destP[i])
_srcP = np.array(_srcP)
_destP = np.array(_destP)
h = compute_homography(_srcP, _destP)
img22 = cv2.warpPerspective(img1, h, (img2.shape[1], img2.shape[0]))
justImg = np.copy(img2)
for j in range(justImg.shape[0]):
    for i in range(justImg.shape[1]):
        if img22[j, i] == 0:
            continue
        else:
            justImg[j, i] = img22[j, i]

h = compute_homography_ransac(srcP[:25], destP[:25], 0.7)
#h /= h[2, 2]
print('2', h)
im1Reg = cv2.warpPerspective(img1, h, (img2.shape[1], img2.shape[0]))
ransacImg = np.copy(img2)
for j in range(ransacImg.shape[0]):
    for i in range(ransacImg.shape[1]):
        if im1Reg[j, i] == 0:
            continue
        else:
            ransacImg[j, i] = im1Reg[j, i]
cv2.imshow('just', img22)
cv2.imshow('just on desk', justImg)
cv2.imshow('ransac', im1Reg)
cv2.imshow('ransac on desk', ransacImg)
hp = cv2.resize(hp, dsize=(img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)
hp = cv2.warpPerspective(hp, h, (img2.shape[1], img2.shape[0]))
hpImg = np.copy(img2)
for j in range(hpImg.shape[0]):
    for i in range(hpImg.shape[1]):
        if hp[j, i] == 0:
            continue
        else:
            hpImg[j, i] = hp[j, i]
cv2.imshow('hp on desk', hpImg)


imgL = cv2.imread('diamondhead-10.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('diamondhead-11.png', cv2.IMREAD_GRAYSCALE)

kpL = orb.detect(imgL, None)
kpL, desL = orb.compute(imgL, kpL)
kpR = orb.detect(imgR, None)
kpR, desR = orb.compute(imgR, kpR)

d1H, d1W = desR.shape
d2H, d2W = desL.shape

ans1 = []
for j1 in range(d1H):
    minCount = 100000
    ansIdx = 0
    for j2 in range(d2H):
        count = 0
        for i1 in range(d1W):
            count += bin(desR[j1][i1]^desL[j2][i1]).count('1')
        if count < minCount:
            ansIdx = j2
            minCount = count
    ans1.append((j1, ansIdx, minCount))
ans1 = sorted(ans1, key=lambda x:x[2])
matches = []
# bf = cv2.BFMatcher()
# matches = bf.match(des1, des2)
# matches = sorted(matches, key = lambda x:x.distance)
#
for item in ans1:
    matches.append(cv2.DMatch(item[0], item[1], item[2]))
#img3 = cv2.drawMatches(imgL, kpL, imgR, kpR, matches[:10], None, flags=2)
cv2.imshow('dd', img3)

srcP = []
destP = []
# for item in matches:
#     srcP.append([kp1[item.queryIdx].pt[0], kp1[item.queryIdx].pt[1]])
#     destP.append([kp2[item.trainIdx].pt[0], kp2[item.trainIdx].pt[1]])
for item in matches:
    srcP.append(kpR[item.queryIdx].pt)
    destP.append(kpL[item.trainIdx].pt)
srcP = np.array(srcP)
destP = np.array(destP)


h = compute_homography_ransac(srcP[:40], destP[:40], 0.8)
landL = cv2.warpPerspective(imgR, h, (imgL.shape[1]+imgR.shape[1], imgL.shape[0]))

avgImgL = np.average(imgL)
avgLandL = np.average(landL)
#landL = landL * (avgImgL/avgLandL)
for j in range(imgL.shape[0]):
    for i in range(imgL.shape[1]):
        landL[j, i] = imgL[j, i]
maxX = 0
for j in range(landL.shape[0]):
    for i in range(landL.shape[1]-1, 1, -1):
        if landL[j, i] != 0:
            if i > maxX:
                maxX = i
            break

landL = landL[:, :maxX]
#landL = cv2.resize(landL, (imgL.shape[1], imgL.shape[0]), interpolation=cv2.INTER_LINEAR)
# kern = np.ones((3, 3))/9
# co = np.copy(landL)
cv2.imshow('landL', landL)
#cv2.imshow('landR', landR)

cv2.waitKey(0)



