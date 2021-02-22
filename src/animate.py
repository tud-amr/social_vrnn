import pathlib
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys, inspect

from descartes import PolygonPatch


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:
    sys.path.insert(0, parentdir)
from src.data_utils.ProcessTrafficData import mergeSegment, LoadTrafficData, FilterTraffic


idx_segments = [145, 147, 148, 152]

path = pathlib.Path(parentdir).absolute()
data_path = path / 'data/real_world/amsterdam_canals'
map_path = data_path / 'canal_map'
dataset = data_path / 'traffic_data.sqlite3'
segment = mergeSegment(idx_segments, map_path)

time_from = datetime(2017, 8, 12, 13)
time_to = datetime(2017, 8, 12, 14)

resolution = [10, 10, .1, np.pi / 48]

traffic_data_raw = LoadTrafficData(dataset, segment, time_from, time_to)
traffic_data = FilterTraffic(traffic_data_raw, segment, resolution)

if False:
    for key in traffic_data.keys():
        dt, x, y, th, vx, vy, w, dim_1, dim_2 = zip(*traffic_data[key])

        x_int = [x[0]]
        y_int = [y[0]]
        shiftx = np.roll(x, -1)
        shifty = np.roll(y, -1)
        vx = shiftx - x
        vy = shifty - y
        vx[-1] = 0
        vy[-1] = 0

        for i in range(len(vx) - 1):
            x_int.append(x_int[i] + 1 * vx[i])

        for i in range(len(vy) - 1):
            y_int.append(y_int[i] + 1 * vy[i])


        fig, ax = plt.subplots()
        route = plt.scatter(x, y, color="red", alpha=0.5, s=0.2)
        ax.add_patch(PolygonPatch(segment, fill=False, alpha=1.0, color='black'))
        plt.savefig(str(key), dpi=400)
        plt.clf()

        fig, ax = plt.subplots()
        route = plt.scatter(x_int, y_int, color="green", alpha=0.5, s=0.2)
        ax.add_patch(PolygonPatch(segment, fill=False, alpha=1.0, color='black'))
        plt.savefig(str(key) + '_interpolated', dpi=400)
        plt.clf()
        exit()

# exit()

# trajectory_index = 0

for i in range(len(traffic_data.keys())):
    key = list(traffic_data.keys())[i]

    dt, x, y, th, vx, vy, w, dim_1, dim_2 = zip(*traffic_data[key])

    speedup = 50

    fig, ax = plt.subplots()
    route = plt.scatter(x, y, c=range(len(x)), alpha=0.5, s=0.5)
    ax.add_patch(PolygonPatch(segment, fill=False, alpha=1.0, color='black'))
    position, = plt.plot(x[0], y[0], 'ro')


    def animate(i):
        position.set_data(x[i], y[i])
        return position,


    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(0, len(x)),
                                        interval=1000 / speedup,
                                        repeat=False)

    plt.show()
