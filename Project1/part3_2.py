import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import scipy.ndimage.filters
import math
from scipy import signal
from skimage import color, data, restoration
from numpy.linalg import inv

def degradation_function(img, func):
    image = np.copy(img)
    # Construct image from blurring function
    for u in range(0, 512):
        for v in range(0, 512):
            image[u,v] = func(u, v)

    return image

def blurr(y,x):
    a = 0.003
    b = 0.003
    T = 1
    C = math.pi*(a*y+b*x)

    if(C == 0):
        return 1

    return (T/C)*math.sin(C)*math.e**(-1j*C)

def toReal(img):
    realImg = np.zeros(img.shape)
    for i in range(0, 512):
        for j in range(0, 512):
            realImg[i,j] = np.absolute(img[i,j])
    return realImg

def normalize(image):
    img = image.copy()
    img = toReal(img)
    img -= img.min()
    img *= 255.0/img.max()
    return img.astype(np.uint8)

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
#################################################### first part

# original image
img = cv2.imread('dew on roses (blurred).tif', 0)
img -= np.amin(img) #map values to the (0, 255) range
img = np.multiply(img, 255.0/np.amax(img))

# 2-DFT
f = np.fft.fft2(img.astype(np.int32)) # 512x512

# centering dft
fft_img = np.fft.fftshift(f)

# for graph of degraded image
magnitude_spectrum = 20*np.log(np.abs(fft_img)) # degraded image G(u,v)
####################################################

# get degradation function
filtered_fft = degradation_function(fft_img, blurr)

f_fft_img = np.fft.ifftshift(filtered_fft)

filtered_img = np.fft.ifft2(f_fft_img)

filtered_img = normalize(filtered_img)

# 2-DFT
ff = np.fft.fft2(filtered_img.astype(np.int32)) # 512x512

# centering dft
ffft_img = np.fft.fftshift(ff)

# for graph of 
magnitude_spectrumf = 20*np.log(np.abs(ffft_img)) # degradation function H(u,v)

######################################################### second part, inverse filtering

inv = fft_img/ffft_img

for i in range(512):
    for j in range(512):
        if math.sqrt((i-256)*(i-256) + (j-256)*(j-256)) >= 80 :
            inv[i][j] = 0

# recentering reverse to fftshift
f_ishift_out = np.fft.ifftshift(inv)

# IDFT
img_back_out = np.fft.ifft2(f_ishift_out)
img_back_out = np.abs(img_back_out)
'''
img_back_out -= np.amin(img_back_out) #map values to the (0, 255) range
img_back_out = np.multiply(img_back_out, 255.0/np.amax(img_back_out))

# 2-DFT
fimg_back_out = np.fft.fft2(img_back_out.astype(np.int32)) # 512x512

# centering dft
fft_imgimg_back_out = np.fft.fftshift(fimg_back_out)

# for graph of degraded image
magnitude_spectrumimg_back_out = 20*np.log(np.abs(fft_imgimg_back_out)) # degraded image G(u,v)
'''
########################################################

# draw graphs
dt = 1
t = np.arange(0, 512, dt)

plt.subplot(2, 2, 1),plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2),plt.plot(t, magnitude_spectrum[200], t, magnitude_spectrumf[200])
plt.title('compare degraded and degradation function'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3),plt.imshow(img_back_out, cmap='gray')
plt.title('Restored Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4),plt.plot(t, magnitude_spectrum[300], t, magnitude_spectrumf[300])
plt.title('compare degraded and degradation function'), plt.xticks([]), plt.yticks([])

plt.show()