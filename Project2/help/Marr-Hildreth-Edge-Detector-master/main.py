import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal
import scipy as sp
from scipy.ndimage import gaussian_filter,laplace, gaussian_laplace
import numpy as np
import cv2
from skimage import exposure, color
import math

def gaussian_kernel(sigma_pixel=1):
	radius = 3 * sigma_pixel # gaussian function spans 3 sigmas
	d = 2 * radius 
	diameter = int(math.ceil(d))
	if(diameter % 2) == 0:
		diameter = diameter + 1.
	ax = np.arange(-diameter // 2 + 1., diameter // 2 + 1.)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-(xx**2 + yy**2) / (diameter))
	return kernel/kernel.sum()

# original image
im = plt.imread('airplane in the sky.tif')
plt.subplot(3, 2, 1),plt.imshow(im, cmap='gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])
img = color.rgb2gray(im)

# Laplacian kernel
laplacian_kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])

# gaussian kernel
sigma = 4
array = gaussian_kernel(sigma)
# laplacian of gaussian
LoG = sp.signal.convolve2d(array, laplacian_kernel, mode='same', boundary ='symm')
plt.subplot(3, 2, 2),plt.imshow(LoG, cmap='gray')
plt.title('LoG'), plt.xticks([]), plt.yticks([])

#LoG = LoG/LoG.sum()
img_LoG = sp.signal.convolve2d(img, LoG, mode='same', boundary = 'symm')
plt.subplot(3, 2, 3),plt.imshow(img_LoG, cmap='gray')
plt.title('LoG og image'), plt.xticks([]), plt.yticks([])
zero_crossings = np.where(np.diff(np.sign(img_LoG)))[0]

minLoG = cv2.morphologyEx(img_LoG, cv2.MORPH_ERODE, np.ones((3,3)))
# 0 threshold
maxLoG1 = 0*cv2.morphologyEx(img_LoG, cv2.MORPH_DILATE, np.ones((3,3)))
zeroCross1 = np.logical_or(np.logical_and(minLoG < 0,  img_LoG > 0), np.logical_and(maxLoG1 > 0, img_LoG < 0))
zeroCross1 = np.multiply(zeroCross1, 255)
plt.subplot(3, 2, 4),plt.imshow(zeroCross1, cmap='gray')
plt.title('Zero crossing 0 threshold'), plt.xticks([]), plt.yticks([])
# 0.04 threshold
maxLoG2 = 0.04*cv2.morphologyEx(img_LoG, cv2.MORPH_DILATE, np.ones((3,3)))
zeroCross2 = np.logical_or(np.logical_and(minLoG < 0,  img_LoG > 0), np.logical_and(maxLoG2 > 0, img_LoG < 0))
zeroCross2 = np.multiply(zeroCross2, 255)
plt.subplot(3, 2, 5),plt.imshow(zeroCross2, cmap='gray')
plt.title('Zero crossing 0.04 threshold'), plt.xticks([]), plt.yticks([])
plt.show()