import matplotlib.pyplot as plt
from skimage import io
from skimage.color import label2rgb
from skimage.segmentation import slic

image = io.imread('baboon_rgb.png')
segmentation = slic(image, n_segments=200,
                    compactness=10, start_label=1)
label = label2rgb(segmentation, image=image, bg_label=0)
plt.imshow(label)
plt.colorbar()
plt.show()