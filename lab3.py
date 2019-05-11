import numpy as np
from scipy import ndimage
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import misc
import cv2

def convolution(X, Y):
	#print(X)
	#print(Y)
	filteredImage = []
	for i in range(len(Y) - len(X) + 1):
		auxRow = []
		for j in range(len(Y[0])-len(X[0]) + 1):
			auxRow.append(matrixProduct(X, Y, i, j))
		filteredImage.append(auxRow)
	return extendMatrix(filteredImage)

def matrixProduct(kernel, matrix, rowNumber, colNumber):
	sum = 0
	for i in range(len(kernel)):
		for j in range(len(kernel[0])):
			sum = sum + kernel[i][j] * matrix[rowNumber+i][colNumber+j]
	return sum

def extendMatrix(matrix):
	newMatrix = []
	auxRow = []
	for i in range(len(matrix[0]) + 2):
		auxRow.append(0)
	newMatrix.append(auxRow)
	for row in matrix:
		auxRow = [0]
		for i in row:
			auxRow.append(i)
		auxRow.append(0)
		newMatrix.append(auxRow)
	auxRow = []
	for i in range(len(matrix[0]) + 2):
		auxRow.append(0)
	newMatrix.append(auxRow)
	return newMatrix
 
def divideMatrix(matrix):
	dividedMatrix = []
	for i in matrix:
		auxList = []
		for j in i:
			j = j/256
			auxList.append(j)
		dividedMatrix.append(auxList)
	#print(dividedMatrix)
	return dividedMatrix

leenaImage = misc.imread('leena512.bmp')

initialKernel = ([[1.0, 4.0, 6.0, 4.0, 1.0],
		 [4.0, 16.0, 24.0, 16.0, 4.0], 
		 [6.0, 24.0, 36.0, 24.0, 6.0], 
		 [4.0, 16.0, 24.0, 16.0, 4.0], 
		 [1.0, 4.0, 6.0, 4.0, 1.0]])

kernel = divideMatrix(initialKernel)

filterIm = convolution(kernel, leenaImage)

plt.subplot(121)
plt.imshow(leenaImage)
plt.title('Original')

plt.subplot(122)
plt.imshow(filterIm)
plt.title('Filtered')
plt.show()

#print(leenaImage)
