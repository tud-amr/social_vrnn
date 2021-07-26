import pathlib

import matplotlib.pyplot as plt
import cv2
import sqlite3
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from descartes import PolygonPatch
from datetime import datetime

from shapely.geometry.polygon import LineString
from shapely.geometry.point import Point

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path: sys.path.insert(0, parentdir)
from src.data_utils.ProcessTrafficData import mergeSegment, LoadTrafficData, GenerateObsmat, \
    createMap, FilterTraffic

# idx_segments = [145, 147, 148, 152]
resolution = [10, 10, .1, np.pi / 48]
idx_segments = range(0, 261)

path = pathlib.Path(__file__).parent.parent.absolute()
data_path = path / 'data/real_world/amsterdam_canals'

map_path = data_path / 'canal_map'
dataset = data_path / 'traffic_data.sqlite3'

segment = mergeSegment(idx_segments, map_path)

time_from = datetime(2017, 8, 12, 13)
time_to = datetime(2017, 8, 12, 14)

# traffic_data_raw = LoadTrafficData(dataset, segment, time_from, time_to)
# traffic_data_filtered = FilterTraffic(traffic_data_raw, segment, resolution)
# traffic_data_filtered = traffic_data_raw
# obsmat = GenerateObsmat(traffic_data_filtered, data_path, save=False)
# createMap(idx_segments, data_path)

canal_map = path / 'data/real_world/amsterdam_canals/canal_map'
with open(canal_map, 'rb') as file_pickle:
    segments = pickle.load(file_pickle)
    
segment = None
for i in segments:
    if segment is None:
        segment = segments[i]
    else:
        segment = segment.union(segments[i])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
fig.set_facecolor('xkcd:white')

ax.set_xlim([segment.bounds[0] - 10, segment.bounds[2] + 10])
ax.set_ylim([segment.bounds[1] - 10, segment.bounds[3] + 10])

## plot canal segment
for sgm_id, sgm in segments.items():
    if sgm_id in range(0, 261):
        ax.add_patch(PolygonPatch(sgm, fill=False, alpha=1.0, color='black'))
        # x, y = sgm.exterior.xy

        # x_mean = (max(x) + min(x))/2
        # y_mean = (max(y) + min(y))/2

        # ax.text(x_mean, y_mean, str(sgm_id), size=2, color='red')

# plt.scatter(x, y, color='limegreen', marker='.', s=5)
# plt.plot(x, y, color='limegreen')
plt.gca().set_aspect('equal')
# plt.axis("off")
plt.tight_layout()

# plt.savefig("segments.pdf", format='pdf')
plt.show()
