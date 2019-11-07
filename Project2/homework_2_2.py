import numpy as np
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
import matplotlib.pyplot as plt
import cv2
from skimage.feature import canny

def hough_transform(img_bin, theta_res=1, rho_res=1):
  nR,nC = img_bin.shape
  theta = np.linspace(-90.0, 0.0, np.ceil(90.0/theta_res) + 1.0)
  theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
 
  D = np.sqrt((nR - 1)**2 + (nC - 1)**2)
  q = np.ceil(D/rho_res)
  nrho = 2*q + 1
  rho = np.linspace(-q*rho_res, q*rho_res, nrho)
  H = np.zeros((len(rho), len(theta)))
  for rowIdx in range(nR):
    for colIdx in range(nC):
      if img_bin[rowIdx, colIdx]:
        for thIdx in range(len(theta)):
          rhoVal = colIdx*np.cos(theta[thIdx]*np.pi/180.0) + rowIdx*np.sin(theta[thIdx]*np.pi/180)
          rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
          H[rhoIdx[0], thIdx] += 1
  return rho, theta, H

# read original image
im = plt.imread('airplane in the sky.tif')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

# read edge image
image = plt.imread('edge004.png')

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Edge image')
ax[0].set_axis_off()
'''
d, theta, h = hough_transform(image, 2, 5)
ax[1].imshow(np.log(1 + h), extent=[(theta[0]), (theta[-1]), d[0], d[-1]], cmap='gray', aspect='auto')
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
'''
lines = probabilistic_hough_line(image, threshold=25, line_length=17, line_gap=8)
black = np.zeros(im.shape)
ax[1].imshow(black, cmap='gray')
summ = 0

for line in lines:
    p0, p1 = line
    print(p0)
    print(p1)
    #if ((p1[1]<350) or (p1[1]>400)):
    #ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), '-b')
    summ+=1
    if summ >25:
      break

x = [382, 482, 562, 885, 805, 1018, 1035, 365, 608]
y = [255, 350, 332, 295, 433, 432, 430, 260, 327]
ax[1].plot(x, y, 'ro')
ax[1].set_xlim((0, im.shape[1]))
ax[1].set_ylim((im.shape[0], 0))
ax[1].set_axis_off()
ax[1].set_title('Possible aircraft body')

plt.tight_layout()
plt.show()