import numpy as np
from scipy import ndimage
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import misc
import cv2
from scipy import fftpack

# Calculate convolution beetween kernel and image (both matrix)
# Return the matrix of the operation
def convolution(kernel, image):
	filteredImage = []
	for i in range(len(image) - len(kernel) + 1):
		auxRow = []
		for j in range(len(image[0])-len(kernel[0]) + 1):
			auxRow.append(matrixProduct(kernel, image, i, j))
		filteredImage.append(auxRow)
	return extendMatrix(filteredImage)

# Calculate the sum of the multiplication of every row and column given
# by the parameters rowNumber and colNumber of
# The matrix kernel and image
# Return total sum
def matrixProduct(kernel, image, rowNumber, colNumber):
	sum = 0
	for i in range(len(kernel)):
		for j in range(len(kernel[0])):
			sum = sum + kernel[i][j] * image[rowNumber+i][colNumber+j]
	return sum

# Generate zeros around matrix variable
# This is for give dark edges to the image
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

# Divide every component of matrix by 256
# This is for the kernel given by the laboratory 
def divideMatrix(matrix):
	dividedMatrix = []
	for i in matrix:
		auxList = []
		for j in i:
			j = j/256
			auxList.append(j)
		dividedMatrix.append(auxList)
	return dividedMatrix

# Apply Gaussian filter given on the image parameter
def gaussianFilter(image):
	initialKernel = ([[1.0, 4.0, 6.0, 4.0, 1.0],
		 			  [4.0, 16.0, 24.0, 16.0, 4.0], 
		 			  [6.0, 24.0, 36.0, 24.0, 6.0], 
		 			  [4.0, 16.0, 24.0, 16.0, 4.0], 
		  			  [1.0, 4.0, 6.0, 4.0, 1.0]])

	kernel = divideMatrix(initialKernel)

	filterIm = convolution(kernel, image)

	generateImageSubPlot('Gaussian Filter', 'Original', 'Gaussian Filter', image, filterIm)

	return filterIm

# Apply Edge Filter given on the image parameter
def edgeFilter(image):
	kernel = [[1, 2, 0, -2, -1], 
			  [1, 2, 0, -2, -1], 
			  [1, 2, 0, -2, -1], 
		 	  [1, 2, 0, -2, -1], 
			  [1, 2, 0, -2, -1]]

	filterIm = convolution(kernel, image)

	generateImageSubPlot('Edge Filter', 'Original', 'Edge Filter', image, filterIm)

	return filterIm

# Function to generate plot 
def generatePlot(figName, figTitle, data):
    plt.figure(figName)
    plt.title(figTitle)
    plt.plot(data)

# Calculate fourier transform of image parameter
def imageFourierTransform(image):	
	f = np.fft.fft2(image)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	return magnitude_spectrum

# Generate two subplot of image1 and image2
def generateImageSubPlot(figName, title1, title2, image1, image2):	
	plt.figure(figName)
	plt.subplot(121)
	plt.imshow(image1, cmap=plt.cm.gray)
	plt.title(title1)
	plt.subplot(122)
	plt.imshow(image2, cmap=plt.cm.gray)
	plt.title(title2)


leenaImage = misc.imread('leena512.bmp')
 
gaussFilter = gaussianFilter(leenaImage)

edgeFilter = edgeFilter(leenaImage)

fourierTransform = imageFourierTransform(leenaImage)

edgeFourierTransform = imageFourierTransform(edgeFilter)

gaussianFourierTransform = imageFourierTransform(gaussFilter)

generateImageSubPlot('Frequency Domain Original Image', 'Original', 'Magnitude Spectrum', leenaImage, fourierTransform)

generateImageSubPlot('Frequency Domain Edge Filter', 'Edge Filter', 'Magnitude Spectrum', edgeFilter, edgeFourierTransform)

generateImageSubPlot('Frequency Domain Gaussian Filter', 'Gaussian Filter',  'Magnitude Spectrum', gaussFilter, gaussianFourierTransform)

plt.show()
