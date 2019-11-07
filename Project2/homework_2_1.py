import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import cv2
from skimage import exposure, color
import math
import scipy.ndimage.filters

def diff(a,b):
	return abs(abs(a)-abs(b))

def zerocrossing(log, threshold):
	zero_crossing = np.zeros(img_LoG.shape)
	for i in range(log.shape[0]-1):
		for j in range(log.shape[1]-1):
			if log[i][j] >= 0:
				# left/right
				if (((log[i][j-1] <= 0 and log[i][j+1] >= 0) or (log[i][j-1] >= 0 and log[i][j+1] <= 0)) and (diff(log[i][j-1], log[i][j+1]) >= threshold)):
					zero_crossing[i][j] = log[i][j]
				# up/down
				elif (((log[i-1][j] <= 0 and log[i+1][j] >= 0) or (log[i-1][j] >= 0 and log[i+1][j] <= 0)) and (diff(log[i+1][j], log[i-1][j]) >= threshold)):
					zero_crossing[i][j] = log[i][j]
				# two diagonals
				elif (((log[i+1][j+1] <= 0 and log[i-1][j-1] >= 0) or (log[i-1][j-1] <= 0 and log[i+1][j+1] >= 0)) and (diff(log[i-1][j-1], log[i-1][j-1]) >= threshold)):
					zero_crossing[i][j] = log[i][j]
				elif (((log[i+1][j-1] <= 0 and log[i-1][j+1] >= 0) or (log[i+1][j-1] >= 0 and log[i-1][j+1] <= 0)) and (diff(log[i-1][j+1], log[i+1][j-1]) >= threshold)):
					zero_crossing[i][j] = log[i][j]
	return zero_crossing

def gaussian_kernel(sigma_pixel=1):
	radius = 3 * sigma_pixel # gaussian function spans 3 sigmas
	d = 2 * radius 
	diameter = int(math.ceil(d))
	if(diameter % 2) == 0:
		diameter = diameter + 1.
	ax = np.arange(-diameter // 2 + 1., diameter // 2 + 1.)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-(xx**2 + yy**2) / (diameter))
	return kernel#/kernel.sum()

# original image
im = plt.imread('airplane in the sky.tif')
plt.subplot(2, 2, 1),plt.imshow(im, cmap='gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])
img = color.rgb2gray(im)
print(img.shape)
# Kernel for negative Laplacian
laplacian_kernel = np.ones((3, 3))*(-1)
laplacian_kernel[1, 1] = 8

# gaussian kernel
sigma = 4
array = gaussian_kernel(sigma)
# laplacian of gaussian
LoG = scipy.ndimage.filters.convolve(array, laplacian_kernel)

img_LoG = sp.signal.convolve2d(img, LoG, mode='same', boundary = 'symm')
plt.subplot(2, 2, 2),plt.imshow(img_LoG, cmap='gray')
plt.title('LoG og image'), plt.xticks([]), plt.yticks([])

# 0 threshold
zeroCross = zerocrossing(img_LoG, 0)
zero_cross0 = zeroCross > 0
zero_cross0 = np.multiply(zero_cross0, 255)
plt.subplot(2, 2, 3),plt.imshow(zero_cross0, cmap='gray')
plt.title('Zero crossing 0% threshold'), plt.xticks([]), plt.yticks([])
cv2.imwrite('edge.png', zero_cross0)

# 0.04 threshold
zeroCross4 = zerocrossing(img_LoG, 0.04*np.amax(img_LoG))
zero_cross4 = zeroCross4 > 0
zero_cross4 = np.multiply(zero_cross4, 255)
plt.subplot(2, 2, 4),plt.imshow(zero_cross4, cmap='gray')
plt.title('Zero crossing 4% threshold'), plt.xticks([]), plt.yticks([])
cv2.imwrite('edge004.png', zero_cross4)

plt.show()