from datetime import datetime

import sqlite3
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from descartes import PolygonPatch

from shapely.geometry.polygon import LineString
from shapely.geometry.point import Point


def mergeSegment(idx_segments, map_path):
    with open(map_path, 'rb') as file_pickle:
        segments = pickle.load(file_pickle)

    segment = None
    for i in idx_segments:
        if segment is None:
            segment = segments[i]
        else:
            segment = segment.union(segments[i])
    return segment


def LoadTrafficData(dataset_path, segment, time_from, time_to):
    """
    Load traffic data between dt_from and dt_to around a given segment
    """

    ## find a bounding box for segment
    x_min = 10 ** 6
    x_max = 0
    y_min = 10 ** 6
    y_max = 0

    x, y = segment.exterior.xy
    x_min = min(x_min, min(x))
    x_max = max(x_max, max(x))
    y_min = min(y_min, min(y))
    y_max = max(y_max, max(y))

    ## read traffic data
    conn_db = sqlite3.connect(str(dataset_path),
                              detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    cur = conn_db.cursor()
    cur.execute(
        "SELECT * FROM traffic_intp WHERE x >= ? and x < ? and y >= ? and y < ? and time >= ? and "
        "time < ? ORDER BY mmsi, time",
        [x_min - 50, x_max + 50, y_min - 50, y_max + 50, time_from, time_to])

    traffic_data = {}
    for data in cur.fetchall():
        key = str(data[0])

        if key in traffic_data.keys():
            traffic_data[key].append(data[1:])

        else:
            traffic_data[key] = [data[1:]]

    conn_db.close()

    return traffic_data


def LoadRawTrafficData(filename, segment, dt_from, dt_to):
    """
    Load (original) traffic data between dt_from and dt_to around given segment
    """

    ## find a bounding box for segment
    x_min = 10 ** 6;
    x_max = 0
    y_min = 10 ** 6;
    y_max = 0

    x, y = segment.exterior.xy
    x_min = min(x_min, min(x))
    x_max = max(x_max, max(x))
    y_min = min(y_min, min(y))
    y_max = max(y_max, max(y))

    ## read traffic data
    conn_db = sqlite3.connect(filename,
                              detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    cur = conn_db.cursor()
    cur.execute(
        "SELECT * FROM traffic WHERE x >= ? and x < ? and y >= ? and y < ? and time >= ? and time < ? ORDER BY mmsi, time",
        [x_min - 50, x_max + 50, y_min - 50, y_max + 50, dt_from, dt_to])

    traffic_data = {}
    for data in cur.fetchall():
        key = str(data[0])

        if key in traffic_data.keys():
            traffic_data[key].append(data[1:])
        else:
            traffic_data[key] = [data[1:]]

    conn_db.close()

    return traffic_data


def FilterTraffic(traffic_data_raw, segment, resolution):
    """
    Filter raw traffic data to remove vessels that are not inside the segment.
    """
    n = 0
    traffic_data_filtered = {}
    for key in traffic_data_raw.keys():
        dt, x, y, th, vx, vy, w, dim_1, dim_2 = zip(*traffic_data_raw[key])

        idx = [i for i, tmp_dt in enumerate(np.diff(dt)) if tmp_dt.seconds > 1]
        idx = [0] + idx
        idx = idx + [-1]

        for i in range(len(idx) - 1):
            x_tmp = x[idx[i] + 1:idx[i + 1]]
            y_tmp = y[idx[i] + 1:idx[i + 1]]
            th_tmp = th[idx[i] + 1:idx[i + 1]]
            N = len(x_tmp)

            if N == 0: continue

            i_begin = 0
            for j in range(N):
                i_begin = j

                if segment.buffer(resolution[0]).intersects(Point(x_tmp[j], y_tmp[j])):
                    break

            i_end = N - 1
            for j in range(N):
                i_end = N - 1 - j

                if segment.buffer(resolution[0]).intersects(
                        Point(x_tmp[N - 1 - j], y_tmp[N - 1 - j])):
                    break

            if i_begin >= i_end: continue

            traj_vessel = LineString(zip(x_tmp[i_begin:i_end + 1], y_tmp[i_begin:i_end + 1]))

            ## check validation -- whether the trajectory is long enough and is not looping around
            if np.sqrt((x_tmp[i_begin] - x_tmp[i_end]) ** 2 + (
                    y_tmp[i_begin] - y_tmp[i_end]) ** 2) < 100 or traj_vessel.length < 100: continue

            traffic_data_filtered[n] = list(zip(dt[idx[i] + 1 + i_begin:idx[i] + 1 + i_end + 1],
                                                x[idx[i] + 1 + i_begin:idx[i] + 1 + i_end + 1],
                                                y[idx[i] + 1 + i_begin:idx[i] + 1 + i_end + 1],
                                                th[idx[i] + 1 + i_begin:idx[i] + 1 + i_end + 1],
                                                vx[idx[i] + 1 + i_begin:idx[i] + 1 + i_end + 1],
                                                vy[idx[i] + 1 + i_begin:idx[i] + 1 + i_end + 1],
                                                w[idx[i] + 1 + i_begin:idx[i] + 1 + i_end + 1],
                                                dim_1[idx[i] + 1 + i_begin:idx[i] + 1 + i_end + 1],
                                                dim_2[idx[i] + 1 + i_begin:idx[i] + 1 + i_end + 1]))

            n += 1
    return traffic_data_filtered


def GenerateObsmat(traffic_data, data_path, save=True):
    """
    Convert filtered traffic data into a obsmat format and optionally save the data as a text file.
    """

    basetime = datetime(year=2017, month=8, day=12)
    obsmat = None

    for key in traffic_data.keys():
        dt, x, y, th, vx, vy, w, dim_1, dim_2 = zip(*traffic_data[key])

        keys = np.full_like(x, fill_value=key)
        zeros = np.zeros_like(x)
        frames = [int((date - basetime).total_seconds()) for date in dt]
        if obsmat is None:
            obsmat = np.vstack((frames, keys, x, zeros, y, vx, zeros, vy)).transpose()
        else:
            stacked = np.vstack((frames, keys, x, zeros, y, vx, zeros, vy)).transpose()
            obsmat = np.vstack((obsmat, stacked))

    obsmat = obsmat[obsmat[:, 0].argsort()]
    if save:
        np.savetxt(data_path / 'obsmat.txt', obsmat, fmt='%e')

    return obsmat


def createMap(idx_segments, data_path, dpi=40):
    canal_map = data_path / 'canal_map'
    with open(canal_map, 'rb') as file_pickle:
        segments = pickle.load(file_pickle)

    segment = None
    for i in segments:
        if segment is None:
            segment = segments[i]
        else:
            segment = segment.union(segments[i])

    segment = None
    for i in idx_segments:
        if segment is None:
            segment = segments[i]
        else:
            segment = segment.union(segments[i])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_facecolor('xkcd:white')

    ax.set_xlim([segment.bounds[0] - 10, segment.bounds[2] + 10])
    ax.set_ylim([segment.bounds[1] - 10, segment.bounds[3] + 10])

    ## plot canal segment
    ax.add_patch(PolygonPatch(segment, alpha=1.0, color='black'))

    plt.gca().set_aspect('equal')
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(data_path / 'map1.png', dpi=dpi, bbox_inches='tight',
                pad_inches=0)

    # Create homography matrix
    H = np.zeros((3, 3))
    H[2][2] = 1

    im = Image.open(data_path / 'map.png')
    x_pixels, y_pixels = im.size

    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    x_width = abs(x_min - x_max)
    y_width = abs(y_min - y_max)

    H[0][0] = x_width / x_pixels
    H[1][1] = y_width / y_pixels

    H[0][2] = x_min
    H[1][2] = y_min

    np.savetxt(data_path / 'H.txt', H, delimiter='  ')
