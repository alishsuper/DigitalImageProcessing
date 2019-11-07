import numpy as N
import scipy.ndimage as I
import matplotlib.image as IM
import matplotlib.pyplot as plt

def hough_transform(img_bin, theta_res=1, rho_res=1):
  nR,nC = img_bin.shape
  theta = N.linspace(-90.0, 0.0, N.ceil(90.0/theta_res) + 1.0)
  theta = N.concatenate((theta, -theta[len(theta)-2::-1]))
 
  D = N.sqrt((nR - 1)**2 + (nC - 1)**2)
  q = N.ceil(D/rho_res)
  nrho = 2*q + 1
  rho = N.linspace(-q*rho_res, q*rho_res, nrho)
  H = N.zeros((len(rho), len(theta)))
  for rowIdx in range(nR):
    for colIdx in range(nC):
      if img_bin[rowIdx, colIdx]:
        for thIdx in range(len(theta)):
          rhoVal = colIdx*N.cos(theta[thIdx]*N.pi/180.0) + rowIdx*N.sin(theta[thIdx]*N.pi/180)
          rhoIdx = N.nonzero(N.abs(rho-rhoVal) == N.min(N.abs(rho-rhoVal)))[0]
          H[rhoIdx[0], thIdx] += 1
  return rho, theta, H

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

# read edge image
image = plt.imread('edge004.png')

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Edge image')
ax[0].set_axis_off()

d, theta, h = hough_transform(image, 2, 5)
print(theta)
ax[1].imshow(N.log(1 + h), extent=[(theta[0]), (theta[-1]), d[0], d[-1]], cmap='gray', aspect='auto')
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')

plt.tight_layout()
plt.show()