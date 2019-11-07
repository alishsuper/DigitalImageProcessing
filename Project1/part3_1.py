import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import scipy.ndimage.filters
import math

img = cv2.imread('dew on roses (noisy).tif', 0)

# create masks
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:200, 100:200] = 255

mask1 = np.zeros(img.shape[:2], np.uint8) 
mask1[200:300, 200:300] = 255

hist = cv2.calcHist([img], [0], mask, [256], [0, 256])

hist1 = cv2.calcHist([img], [0], mask1, [256], [0, 256])
sumpixel = 0
for i in range(255):
    sumpixel = sumpixel + hist[i]

mean = 0
for i in range(255):
    mean = mean + i*hist[i]/sumpixel
print('mean', mean)

variance = 0
for i in range(255):
    variance = variance + (i-mean)**2*hist[i]/sumpixel
print('variance', variance)
print(math.sqrt(variance))

#########################################################
sumpixel1 = 0
for i in range(255):
    sumpixel1 = sumpixel1 + hist1[i]

mean1 = 0
for i in range(255):
    mean1 = mean1 + i*hist1[i]/sumpixel1
print('mean1', mean1)

variance1 = 0
for i in range(255):
    variance1 = variance1 + (i-mean1)**2*hist1[i]/sumpixel1
print('variance1', variance1)
print(math.sqrt(variance1))

blur = cv2.GaussianBlur(img, (5,5), 0)

plt.subplot(2, 2, 1),plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2), plt.plot(hist)
plt.title('Histogram'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.plot(hist1)
plt.title('Histogram'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4),plt.imshow(blur, cmap='gray')
plt.title('Reconstructed image'), plt.xticks([]), plt.yticks([])

plt.show()