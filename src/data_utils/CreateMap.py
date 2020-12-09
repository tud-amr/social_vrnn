import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from descartes import PolygonPatch


filename = "../../data/real_world/amsterdam_canals/canal_map"
with open(filename, 'rb') as file_pickle:
    segments = pickle.load(file_pickle)

segment = None
for i in segments:
    if segment is None:
        segment = segments[i]
    else:
        segment = segment.union(segments[i])

idx_segments = [145, 147, 148, 152]
segment = None
for i in idx_segments:
    if segment is None:
        segment = segments[i]
    else:
        segment = segment.union(segments[i])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
fig.set_facecolor('xkcd:white')

## plot canal segment
ax.add_patch(PolygonPatch(segment, alpha=1.0, color='black'))

plt.axis("equal")
plt.axis("off")
fig.savefig('../../data/real_world/amsterdam_canals/map.png', dpi=20, bbox_inches='tight', pad_inches=0)
# plt.show()

# Create homography matrix
H = np.zeros((3, 3))
H[2][2] = 1

im = Image.open('../../data/real_world/amsterdam_canals/map.png')
x_pixels, y_pixels = im.size

y_min, y_max = ax.get_ylim()
x_min, x_max = ax.get_xlim()
x_width = abs(x_min - x_max)
y_width = abs(y_min - y_max)

H[0][0] = x_width / x_pixels
H[1][1] = -y_width / y_pixels

H[0][2] = x_min
H[1][2] = y_max

np.savetxt('../../data/real_world/amsterdam_canals/H.txt', H, delimiter='  ')
