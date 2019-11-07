import numpy as np

from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm

# Line finding using the Probabilistic Hough Transform
#image = data.camera()
image = plt.imread('edge004.png')
edges = canny(image)#, 2, 1, 25)
lines = probabilistic_hough_line(image, threshold=10, line_length=30, line_gap=3)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, 'gray')
ax[0].set_title('Input image')

ax[1].imshow(edges, 'gray')
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0, 'gray')
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]), '-b')
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()