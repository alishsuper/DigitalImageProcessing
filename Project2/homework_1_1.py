import cv2
from matplotlib import pyplot as plt

# a - orginal image
img = cv2.imread('dew on roses (color).tif')
# get r,g,b
r = img[:, :, 2] # get r components
g = img[:, :, 1]
b = img[:, :, 0]
rgb_img = cv2.merge([r, g, b])     # switch it to rgb

plt.subplot(2, 2, 1),plt.imshow(rgb_img)
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2),plt.imshow(r, 'gray')
plt.title('Red'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3),plt.imshow(g, 'gray')
plt.title('Green'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4),plt.imshow(b, 'gray')
plt.title('Blue'), plt.xticks([]), plt.yticks([])

plt.show()
