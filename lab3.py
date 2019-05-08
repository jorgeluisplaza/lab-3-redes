import numpy as np
import scipy as sc



def convolution():
	#sum = 0
	X = [[12, 7, 3], [4, 6, 7], [7, 9, 2]]
	Y = [[12, 5, 6, 7, 9], [5, 6, 3, 2, 1], [8, 9, 2, 3 ,5]]
	filteredImage = []
	for i in range(len(Y) - len(X) + 1):
		auxRow = []
		for j in range(len(Y[0])-len(X[0]) + 1):
			auxRow.append(matrixProduct(X, Y, i, j))
		filteredImage.append(auxRow)
	return extendMatrix(filteredImage)


def defineColumn(matrix, colNumber):
	auxColumn = []
	for row in matrix:
		auxColumn.append(row[colNumber])
	return auxColumn


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
			auxRow.append(ic)
		auxRow.append(0)
		newMatrix.append(auxRow)
	auxRow = []
	for i in range(len(matrix[0]) + 2):
		auxRow.append(0)
	newMatrix.append(auxRow)
	return newMatrix

print(convolution())
