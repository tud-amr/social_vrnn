from data_utils.ProcessTrafficData import *
import pathlib

buf = -5.0  # the canal segment map to add safety margin in path planning
resolution = [10, 10, .1, np.pi / 40]

# idx_segments = [152]
# idx_segments = [145, 147, 148, 152]
idx_segments = [194, 149, 148, 257, 152, 259, 145, 144, 147, 72, 65, 69, 96, 76, 74, 80, 77]

path = pathlib.Path(__file__).parent.parent.absolute()
data_path = path / 'data/real_world/amsterdam_canals'

map_path = data_path / 'canal_map'
dataset = data_path / 'traffic_data.sqlite3'


segment = mergeSegment(idx_segments, map_path)

time_from = datetime(2017, 8, 12, 13)
time_to = datetime(2017, 8, 15, 14)

traffic_data_raw = LoadTrafficData(dataset, segment, time_from, time_to)
print("Traffic data loaded")
traffic_data_filtered = FilterTraffic(traffic_data_raw, segment, resolution)
print("Data filtered")
obsmat = GenerateObsmat(traffic_data_filtered, data_path, save=True)
print("Obsmat done")
createMap(idx_segments, data_path)

exit()


canal_map = '/home/jitske/Documents/Dataset/canal_map'
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


x = obsmat[:, 2]
y = obsmat[:, 4]


fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_facecolor('xkcd:white')

ax.set_xlim([segment.bounds[0] - 10, segment.bounds[2] + 10])
ax.set_ylim([segment.bounds[1] - 10, segment.bounds[3] + 10])

## plot canal segment

ax.add_patch(PolygonPatch(segment, fill=False, alpha=1.0, color='black'))
plt.scatter(x, y, color='limegreen', marker='.', s=1)
plt.gca().set_aspect('equal')
# plt.axis("off")
plt.tight_layout()
plt.show()

fig.savefig('map.png', dpi=400, bbox_inches='tight',
            pad_inches=0)


createMap(idx_segments, data_path)
