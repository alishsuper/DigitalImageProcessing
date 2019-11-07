import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import scipy.ndimage.filters
from scipy.signal import convolve as convolvesig

# a - orginal image
img = cv2.imread('Capture.tif', 0)
img -= np.amin(img) #map values to the (0, 255) range
img = np.multiply(img, 255.0/np.amax(img))

# b - Laplacian
# Kernel for negative Laplacian
kernel = np.ones((3,3))*(-1)
kernel[1,1] = 8

# Convolution of the image with the kernel:
lap = scipy.ndimage.filters.convolve(img, kernel)

#Map Laplacian to some new range:
laps = lap*100.0/np.amax(lap) #Sharpening factor!
print('Scaled laps  ', np.amax(laps), np.amin(laps))

# Map Laplacian to the (0, 255) range:
lap -= np.amin(lap)
lap *= 255.0/np.amax(lap)
print('Scaled lap  ', np.amax(lap), np.amin(lap))

# c - Sharpened Image
sharpened_image = img + laps
print('SharpImg    ', np.amax(sharpened_image), np.amin(sharpened_image))

sharpened_image = abs(sharpened_image) #Get rid of negative values
print('SharpImg abs', np.amax(sharpened_image), np.amin(sharpened_image))

sharpened_image *= 255.0/np.amax(sharpened_image)
print('sharpened_image', np.amax(sharpened_image), np.amin(sharpened_image))

# d - Sobel Gradient
##########
res1 = np.array([ [-1, -2, -1], 
[0, 0, 0], 
[1, 2, 1]])
res2 = np.array([ [5, 5, 5], 
[-3, 0, -3], 
[-3, -3, -3]])
##########
sobelx = scipy.ndimage.filters.convolve(img, res1)#cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = scipy.ndimage.filters.convolve(img, res2)#cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

sobel_gradient = abs(sobely)#abs(sobelx) + abs(sobely)

plt.subplot(2,2,1),plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2),plt.imshow(lap, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4),plt.imshow(sobel_gradient, cmap='gray')
plt.title('Sobel Gradient'), plt.xticks([]), plt.yticks([])

plt.show()