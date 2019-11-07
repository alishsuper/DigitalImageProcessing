import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import scipy.ndimage.filters
import math

# original image
img = cv2.imread('tulips irises.tif', 0)
img -= np.amin(img) #map values to the (0, 255) range
img = np.multiply(img, 255.0/np.amax(img))

# zero padding
img = np.pad(img, ((0, 512), (0, 512)), 'constant')

# 2-DFT
f = np.fft.fft2(img) # 1024x1024

# DFT centering, F'(u,v)
fshift = np.fft.fftshift(f) # 1024x1024

original_fshift = np.fft.fftshift(f)

# for graph
magnitude_spectrum = 20*np.log(np.abs(fshift))

rows, cols = img.shape # 1024, 1024
crow, ccol = int(rows/2), int(cols/2) # 512, 512

# TODO: inside rectangular make equals zero
for i in range(2*crow):
    for j in range(2*ccol):
        if math.sqrt((i-512)*(i-512) + (j-512)*(j-512)) <= 60 :
            fshift[i][j] = 0

# G2(u,v), IDFT (1024*1024) recentering reverse to fftshift
f_ishift = np.fft.ifftshift(fshift)

# IDFT
img_back = np.fft.ifft2(f_ishift)

img_back = np.abs(img_back)

# TODO: outside equals zero
print(np.amax(original_fshift))
for i in range(2*crow):
    for j in range(2*ccol):
        if math.sqrt((i-512)*(i-512) + (j-512)*(j-512)) >= 60 :
            original_fshift[i][j] = 0

# G2(u,v), IDFT (1024*1024) recentering reverse to fftshift
f_ishift_out = np.fft.ifftshift(original_fshift)

# IDFT
img_back_out = np.fft.ifft2(f_ishift_out)

img_back_out = np.abs(img_back_out)

# croping images
crop_img_back = img_back[0:512, 0:512]

crop_img_back_out = img_back_out[0:512, 0:512]

# draw graphs
plt.subplot(2, 2, 1),plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2),plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('DFT center'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3),plt.imshow(crop_img_back, cmap='gray')
plt.title('IDFT inside zero, HPF'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4),plt.imshow(crop_img_back_out, cmap='gray')
plt.title('IDFT outside zero, smoothing'), plt.xticks([]), plt.yticks([])

plt.show()