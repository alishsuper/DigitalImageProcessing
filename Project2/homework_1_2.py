import cv2
from matplotlib import pyplot as plt

img = cv2.imread('dew on roses (color).tif')

# get r,g,b
r = img[:, :, 2]
g = img[:, :, 1]
b = img[:, :, 0]
rgb_img_original = cv2.merge([r, g, b])     # switch it to rgb
print(b)
# modify blue component
b = cv2.equalizeHist(b)
print(b)
rgb_img = cv2.merge([r,g,b])     # switch it to rgb

plt.subplot(1, 2, 1),plt.imshow(rgb_img_original)
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2),plt.imshow(rgb_img)
plt.title('Modify blue component'), plt.xticks([]), plt.yticks([])

plt.show()
