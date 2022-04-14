import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np

data_number = '000009'

im = Image.open(f'/media/cinnes/Storage/Datasets/KITTI/ObjectDetection2D/LeftCam/training/{data_number}.png')
with open(f'/media/cinnes/Storage/Datasets/KITTI/ObjectDetection2D/Labels/training/{data_number}.txt') as f:
    labels = []
    for l in f.readlines():
        labels.append(l[:-1].split(' '))

labels = np.array(labels)



print("Full labels")
print(labels)

plt.imshow(im)

ax = plt.gca()

print("bounds")
for i in range(len(labels)):
    class_name = labels[i][0]
    if class_name != "DontCare":
        xmin, ymin, xmax, ymax = labels[i][4:8].astype('float')
        # print(bounds)
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.annotate(class_name, xy=(xmin, ymin), color='r')

plt.show()
