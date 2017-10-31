# -*- coding: utf-8 -*-

from numpy import *
import random


def loadDataSet(fileName):  # 构建数据库和标记库
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))  # 只有一列
    return dataMat, labelMat


def selectJrand(i, m):  # 生成一个随机数
    j = i
    while (j == i):
        j = int(random.uniform(0, m))  # 生成一个[0, m]的随机数，int转换为整数。注意，需要import random
    return j


def clipAlpha(aj, H, L):  # 阈值函数
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn);
    labelMat = mat(classLabels).transpose()
    b = 0;
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):  # 迭代次数
        alphaPairsChanged = 0
        for i in range(m):  # 在数据集上遍历每一个alpha
            # print alphas
            # print labelMat
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # fXi=float(np.multiply(alphas, labelMat).T*dataMatrix*dataMatrix[i, :].T)+b  #.T也是转置
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)  # 从m中选择一个随机数，第2个alpha j
                fXj = float(multiply(alphas, labelMat).T * dataMatrix * dataMatrix[j, :].T) + b
                Ej = fXj - float(labelMat[j])

                alphaIold = alphas[i].copy()  # 复制下来，便于比较
                alphaJold = alphas[j].copy()

                if (labelMat[i] != labelMat[j]):  # 开始计算L和H
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L==H')
                    continue

                    # eta是alphas[j]的最优修改量，如果eta为零，退出for当前循环
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('eta>=0')
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta  # 调整alphas[j]
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):  # 如果alphas[j]没有调整
                    print('j not moving enough')
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # 调整alphas[i]
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1

                print('iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas


if __name__ == "__main__":
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

    print(b, alphas)