import matplotlib.pyplot as plt
from skimage import filters
from skimage import exposure
import cv2
import numpy as np
import numba

def diff_mean_sq(mk, mg):
    return mk*mk - 2*mk*mg + mg*mg

@numba.jit
def prob(start, stop):
    summ = 0
    for i in range(start, stop+1):
        summ = summ + hist[i]/sum(hist)
    return summ

@numba.jit
def mean(start, stop):
    summ = 0
    for i in range(start, stop+1):
        summ = summ + i * hist[i]/sum(hist)
    return summ

img = cv2.imread('dew on roses.tif', 0)
val = filters.threshold_otsu(img)
'''
hist, bins_center = exposure.histogram(img)
print(hist)
print(sum(hist))
print(hist[227])
'''
hist, bins = np.histogram(img.ravel(),256,[0,256])
#print(hist) # n[i]
print(sum(hist)) # MN
summ = 0
L = 256 + 1
for i in range(L-1):
    summ = summ + hist[i]/sum(hist) # p[i]
print(summ)
print('start')
start = 2
variance = np.zeros(img.shape)
P1 = 0
P2 = 0
P3 = 0
m1 = 0
m2 = 0
m3 = 0
mg = 0
optimum = 0
opt_k1 = 0
opt_k2 = 0
'''
for k1 in range(1, L-3): # k1
    for k2 in range(start, L-2): # k2
        # P1, P2, P3
        P1 = prob(0, k1)
        P2 = prob(k1+1, k2)
        P3 = prob(k2+1, 255)
        # m1, m2, m3
        if P1 == 0:
            break
        else:
            m1 = 1/P1 * mean(0, k1)
        if P2 == 0:
            break
        else:
            m2 = 1/P2 * prob(k1+1, k2)
        if P3 == 0:
            break
        else:
            m3 = 1/P3 * prob(k2+1, 255)
        # global mean, mg
        mg = P1 * m1 + P2 * m2 + P3 * m3
        print('k1', k1, 'k2', k2)
        variance[k1][k2] = P1 * diff_mean_sq(m1, mg) + P2 * diff_mean_sq(m2, mg) + P3 * diff_mean_sq(m3, mg)
        if optimum < variance[k1][k2]:
            optimum = variance[k1][k2]
            opt_k1 = k1
            opt_k2 = k2
    start = start + 1
'''
print('optimal')
print(optimum, np.amax(variance), variance[110][160])
print(opt_k1)
print(opt_k2)

plt.figure(figsize=(9, 4))
plt.subplot(121)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')

fig1 = img <= 110
plt.subplot(122), plt.plot(hist)
#plt.imshow(hist, cmap='gray', interpolation='nearest')
#plt.axis('off')
'''
fig2 = np.logical_and(img > 110,  img <= 160)
plt.subplot(153)
plt.imshow(fig2, cmap='gray', interpolation='nearest')
plt.axis('off')

fig3 = img > 160
plt.subplot(154)
plt.imshow(fig3, cmap='gray', interpolation='nearest')
plt.axis('off')

for i in range(512):
    for j in range(512):
        if img[i][j] <= 110:
            img[i][j] = 0
        if (img[i][j] > 110) and (img[i][j] <= 160):
            img[i][j] = 127
        if img[i][j] > 160:
            img[i][j] = 255
           
plt.subplot(155)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')
'''
plt.tight_layout()
plt.show()