import matplotlib.pylab as plt
from skimage import io
from skimage.morphology import binary_opening
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches

image = io.imread('circles.png')
image = image[:,:,1]
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i, j] == 0 or image[i, j] == 102:
            image[i, j] = 255
        else:
            image[i, j] = 0
image = binary_opening(image)
img_label = label(image)
regions = regionprops(img_label)
fig, ax = plt.subplots(figsize=(7, 4))
ax.imshow(img_label, cmap='gray')
for i in regions:
    param = i.eccentricity
    if param > 0.1:
        minr, minc, maxr, maxc = i.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)
ax.set_axis_off()
plt.tight_layout()
plt.show()

