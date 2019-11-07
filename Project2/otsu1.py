import matplotlib.pyplot as plt
from skimage import filters
from skimage import exposure
import cv2
import numpy as np
from skimage.exposure import histogram

img = cv2.imread('dew on roses.tif', 0)
val = filters.threshold_otsu(img)
#mult = skimage.filters.threshold_multiotsu
print(val)
'''
hist, bins_center = exposure.histogram(img)

plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')

# otsu threshold
plt.subplot(132)
plt.imshow(img < val, cmap='gray', interpolation='nearest')
plt.axis('off')

# histogram
plt.subplot(133)
plt.plot(bins_center, hist, lw=2)
plt.axvline(val, color='k', ls='--')

plt.tight_layout()
plt.show()
'''