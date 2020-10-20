import matplotlib.pylab as plt
from skimage import data
from skimage.filters import threshold_otsu
# from  skimage import io
import numpy as np

image = data.coins()
# image = io.imread('rice.tif')[50:-50, 50:-50]
image1 = np.array(image, copy=True)
image2 = np.array(image, copy=True)
my_thresh = 110
otsu_thresh = threshold_otsu(image)

fig, axes = plt.subplots(ncols=4, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 4, 1)
ax[1] = plt.subplot(1, 4, 2)
ax[2] = plt.subplot(1, 4, 3)
ax[3] = plt.subplot(1, 4, 4, sharex=True, sharey=True)
ax[0].imshow(image, cmap='gray')
ax[0].set_title('original img')
ax[0].axis('off')
ax[1].hist(image.ravel(), bins=256)
ax[1].axvline(my_thresh, color='r')
ax[1].axvline(otsu_thresh, color='b')
ax[1].set_title('histogram')

for i in range(image1.shape[0]):
    for j in range(image1.shape[1]):
        if image1[i, j] < my_thresh:
            image1[i, j] = 0
        else:
            image1[i, j] = 255
ax[2].imshow(image1, cmap='gray')
ax[2].set_title('my_thresh')
ax[2].axis('off')

for i in range(image2.shape[0]):
    for j in range(image2.shape[1]):
        if image2[i, j] < otsu_thresh:
            image2[i, j] = 0
        else:
            image2[i, j] = 255
ax[3].imshow(image2, cmap='gray')
ax[3].set_title('otsu_thresh')
ax[3].axis('off')
plt.show()

