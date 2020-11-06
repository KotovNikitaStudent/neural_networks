import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

img = io.imread('pentagon.tif')
edges = canny(img, low_threshold=50, high_threshold=200)
test_angles = np.linspace(-np.pi, np.pi, 360)
h, theta, d = hough_line(edges, theta=test_angles)
fig, axes = plt.subplots(1, 2)
ax = axes.ravel()
ax[0].imshow(edges, cmap='gray')
ax[0].set_title('canny')
ax[0].set_axis_off()
ax[1].imshow(edges, cmap='gray')
origin = np.array((0, edges.shape[1]))
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax[1].plot(origin, (y0, y1), 'b')
ax[1].set_xlim(origin)
ax[1].set_ylim((edges.shape[0], 0))
ax[1].set_axis_off()
ax[1].set_title('hough transform')
plt.tight_layout()
plt.show()

