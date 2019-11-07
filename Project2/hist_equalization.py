import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dew on roses (color).tif')

# get r,g,b
r = img[:, :, 2]
g = img[:, :, 1]
b = img[:, :, 0]

hist, bins = np.histogram(b.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[b]

plt.plot(cdf_normalized, color = 'b')
plt.hist(b.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()