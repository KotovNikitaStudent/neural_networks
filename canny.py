import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage import io

image = io.imread('peppers_rgb.png')[:,:,0]
fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 2, 1)
ax[1] = plt.subplot(1, 2, 2, sharex=ax[0], sharey=ax[0])
ax[0].imshow(image, cmap='gray')
ax[0].set_title('original')
ax[0].axis('off')
border = canny(image, low_threshold=20, high_threshold=100)
ax[1].imshow(border, cmap='gray')
ax[1].set_title('canny')
plt.show()
