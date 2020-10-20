import matplotlib.pylab as plt
from skimage import data
from skimage.measure import label, regionprops

image = data.coins()
image1 = image > 107
label_img = label(image1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(label_img, cmap='gray')
j = 1
for i in regionprops(label_img):
    if i.area >= 107:
        print(f'\nколичество пикселей сегментированной {j}-й области: ', i.area)
        print(f'эксцентриситет {j}-й области: ', i.eccentricity)
        print(f'длина большой полуоси эллипса {j}-й области: ', i.major_axis_length)
        print(f'длина малой полуоси эллипса {j}-й области: ', i.minor_axis_length)
        print(f'периметр {j}-й области: ', i.perimeter)
        j = j + 1
ax.set_axis_off()
plt.tight_layout()
plt.show()