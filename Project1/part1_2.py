import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import scipy.ndimage.filters

# a - orginal image
img = cv2.imread('dew on roses (noisy).tif', 0)
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

# d - Sobel Gradient
##########
res1 = np.array([ [-1, -2, -1], 
[0, 0, 0], 
[1, 2, 1]])
res2 = np.array([ [-1, 0, 1], 
[-2, 0, 2], 
[-1, 0, 1]])
##########
sobelx = scipy.ndimage.filters.convolve(img, res1)#cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = scipy.ndimage.filters.convolve(img, res2)#cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

sobel_gradient = abs(sobelx) + abs(sobely)

# e - Smoothed image
kernel_box = np.ones((5,5), np.float32)
smoothed_sobel = cv2.filter2D(sobel_gradient, -1, kernel_box)

# f - Mask image
mask_image = cv2.multiply(smoothed_sobel, laps)
mask_image -= np.amin(mask_image)
mask_image *= 255.0/np.amax(mask_image)
print('mask abs', np.amax(mask_image), np.amin(mask_image))

# g - Sharpened image
sharpened_image = mask_image + img
print('sharpened_image', np.amax(sharpened_image), np.amin(sharpened_image))

# h - Final result
normalized_image = sharpened_image/np.amax(sharpened_image)
print('normalized_image', np.amax(normalized_image), np.amin(normalized_image))
power_law = 255 * abs(normalized_image) ** 0.5

print('power_law', np.amax(power_law), np.amin(power_law))

plt.subplot(2,2,1),plt.imshow(smoothed_sobel, cmap='gray')
plt.title('Smoothed Sobel'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2),plt.imshow(mask_image, cmap='gray')
plt.title('Mask Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4),plt.imshow(power_law, cmap='gray')
plt.title('Power-law'), plt.xticks([]), plt.yticks([])

plt.show()