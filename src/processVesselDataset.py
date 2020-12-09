from src.data_utils.ProcessTrafficData import *
import pathlib

# buf = -5.0  # the canal segment map to add safety margin in path planning
resolution = [10, 10, .1, np.pi / 40]

# idx_segments = [152]
idx_segments = [145, 147, 148, 152]

path = pathlib.Path(__file__).parent.parent.absolute()
data_path = path / 'data/real_world/amsterdam_canals'

map_path = data_path / 'canal_map'
dataset = data_path / 'traffic_data.sqlite3'

segment = mergeSegment(idx_segments, map_path)

time_from = datetime(2017, 8, 12, 13)
time_to = datetime(2017, 8, 12, 14)

traffic_data_raw = LoadTrafficData(dataset, segment, time_from, time_to)
traffic_data_filtered = FilterTraffic(traffic_data_raw, segment, resolution)
GenerateObsmat(traffic_data_filtered, data_path)
createMap(idx_segments, data_path)
