import matplotlib.pylab as plt
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening
import numpy as np
from skimage import io

image = io.imread('noisy_fingerprint.tif')
image1 = np.array(image, copy=True)
image2 = np.array(image, copy=True)
image3 = np.array(image, copy=True)
image4 = np.array(image, copy=True)

fig, axes = plt.subplots(ncols=5, figsize=(17, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 5, 1)
ax[1] = plt.subplot(1, 5, 2)
ax[2] = plt.subplot(1, 5, 3)
ax[3] = plt.subplot(1, 5, 4)
ax[4] = plt.subplot(1, 5, 5, sharex=True, sharey=True)

ax[0].imshow(image, cmap='gray')
ax[0].set_title('original')
ax[0].axis('off')
image1 = binary_erosion(image)
ax[1].imshow(image1, cmap='gray')
ax[1].set_title('erosion')
image2 = binary_dilation(image)
ax[2].imshow(image2, cmap='gray')
ax[2].set_title('dilation')
image3 = binary_opening(image)
ax[3].imshow(image3, cmap='gray')
ax[3].set_title('opening')
image4 = binary_closing(image)
ax[4].imshow(image4, cmap='gray')
ax[4].set_title('closing')
plt.show()
