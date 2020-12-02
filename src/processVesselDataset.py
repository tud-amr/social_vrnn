from datetime import datetime
import numpy as np
import pickle

from src.data_utils.ProcessTrafficData import LoadTrafficData, FilterTraffic, GenerateObsmat

buf = -5.0  # the canal segment map to add safety margin in path planning
resolution = [10, 10, .1, np.pi / 40]

# list of segment indices (and its neighboring segments) for experiments
idx_segments = [145, 147, 148, 152]  # indices for segments to select for path planning
# idx_neighbors = [113, 125, 130, 134, 135, 136]

filename = "/Users/tuhindas/Documents/Tuhin/Computer Science/Year 3/Roboat/social_traj_planning/data/canal_map"
with open(filename, 'rb') as file_pickle:
    segments = pickle.load(file_pickle)

segment = None
for i in idx_segments:
    if segment == None:
        segment = segments[i]
    else:
        segment = segment.union(segments[i])

time_from = datetime(2017, 8, 12, 13)
time_to = datetime(2017, 8, 12, 14)

# TODO upload data and add path to data
filename = "/Users/tuhindas/Documents/Tuhin/Computer Science/Year 3/Roboat/social_traj_planning/data/traffic_data.sqlite3"
traffic_data_raw = LoadTrafficData(filename, segment, time_from, time_to)

# remove the traffic data outside the segment
traffic_data_filtered = FilterTraffic(traffic_data_raw, segment, resolution)

GenerateObsmat(traffic_data_filtered)