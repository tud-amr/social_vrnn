import numpy as np
import pylab as pl
import math
import sys
import os

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import cv2 as cv
sys.path.append('../../src')

def compute_radial_distance_vector(number_elements, relative_positions, max_range=10.0, min_angle=0, max_angle=2*np.pi, normalize=False):
  """
  Compute the distance to surrounding objects in a radially discretized way.
  For each element there will be a floating point distance to the closest object in this sector.
  This allows to preserve the continuous aspect of the distance vs. a standard grid.

  !!! Attention: 0 angle is at the negative x-axis.

  number_elements: radial discretization
  relative_positions: relative positions of the surrounding objects in the local frame (numpy array)
  max_range: maximum range of the distance measurement

  returns:
  radial_dist_vector: contains the distance to the closest object in each sector
  """
  radial_dist_vector = max_range * np.ones([number_elements])
  radial_resolution = (max_angle-min_angle) / float(number_elements)
  if np.any(np.isnan(relative_positions)):
    print("inf")
  for ii in range(relative_positions.shape[0]):
    phi = math.atan2(relative_positions[ii, 1], relative_positions[ii, 0]) + np.pi
    rad_idx = int((phi - min_angle) / radial_resolution)
    if rad_idx >= 0 and rad_idx < number_elements:
      radial_dist_vector[rad_idx] = min(radial_dist_vector[rad_idx], np.linalg.norm(relative_positions[ii,:]))
  if normalize:
    radial_dist_vector /= max_range

  return radial_dist_vector

def sigmoid(x):
  return 1.0/(1.0+np.exp(-x))

def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def positions_in_local_frame(ego_pos, heading, other_pos):
  proj_matrix = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
  local_frame_pos = np.zeros([0, 2])

  for ii in range(other_pos.shape[0]):
    rel_pos = other_pos[ii, :] - ego_pos
    local_frame_pos = np.append(local_frame_pos, np.expand_dims(np.matmul(proj_matrix, rel_pos), axis=0), axis=0)

  return local_frame_pos

def rotate_grid_around_center(grid, angle):
    """
    inputs:
      grid: numpy array (gridmap) that needs to be rotated
      angle: rotation angle in degrees
    """
    # Rotate grid into direction of initial heading
    grid = grid.copy()
    rows, cols = grid.shape
    M = cv2.getRotationMatrix2D(center=(rows / 2, cols / 2), angle=angle, scale=1)
    grid = cv2.warpAffine(grid, M, (rows, cols))

    return grid

def rotate_batch(batch_y, batch_x):
  """
    inputs:
      batch_y: trainning velocities batch containing vx, vy on the global frame
      batch_x: trainning or prediction state batch containing x,y, vx, vy on the global frame
    """
  # rotate initial vx and vy in global frame to quary-agent local frame
  by = batch_y.copy()  # rotated output batch values
  heading = math.atan2(batch_x[1], batch_x[0])
  rot_mat = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
  for pred_step in range(by.shape[0]):
    by[pred_step,:] = np.dot(rot_mat, by[pred_step,:])
  return by

def rotate_batch_to_local_frame(batch_y, batch_x):
  """
    inputs:
      batch_y: trainning velocities batch containing vx, vy on the global frame
      batch_x: trainning or prediction state batch containing x,y, vx, vy on the global frame
    """
  # rotate initial vx and vy in global frame to quary-agent local frame
  bx = batch_x.copy()  # rotated input batch values
  by = batch_y.copy()  # rotated output batch values
  for batch_idx in range(batch_x.shape[0]):
    for tbp_step in range(batch_x.shape[1]):
      heading = math.atan2(bx[batch_idx, tbp_step, 3], bx[batch_idx, tbp_step, 2])
      rot_mat = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
      bx[batch_idx, tbp_step, 2:] = np.dot(rot_mat, bx[batch_idx, tbp_step, 2:])
      for pred_step in range(int(by.shape[2] / 2)):
        by[batch_idx, tbp_step, 2 * pred_step:2 * pred_step + 2] = np.dot(rot_mat, by[batch_idx, tbp_step,
                                                                                   2 * pred_step:2 * pred_step + 2])
  return bx , by

def plot_grid(ax, center, grid, grid_resolution, submap_size):
    """
    Plot a binary occupancy grid in the axis ax.
    Specify center, resolution and size in order to align it with the coordinate frame.
    """
    plot_values = np.zeros([0, 2])
    for idx_x in range(grid.shape[0]):
        for idx_y in range(grid.shape[1]):
            grid_coordinate_local = np.array([idx_x, idx_y]) * grid_resolution - submap_size / 2.0
            grid_coordinate_global = grid_coordinate_local + center
            if grid[idx_x, idx_y] == 1:
                plot_values = np.append(plot_values, np.array([[grid_coordinate_global[0], grid_coordinate_global[1]]]),
                                        axis=0)
    return ax.plot(plot_values[:, 0], plot_values[:, 1], marker='s', alpha=0.8, color='k', lw=0)

def create_map_from_png(file_name,resolution,map_size,map_center,data_path='../data',):
  # Create grid for SF model
  print("create_map_from_png")
  grid_map = {}
  grid_map['Resolution'] = resolution
  grid_map['Size'] = map_size  # map size in [m]
  grid_map['Map'] = np.zeros((int(grid_map['Size'][0] / grid_map['Resolution']),
                              int(grid_map['Size'][1] / grid_map['Resolution'])))

  #map_center = grid_map['Size'] / 2  # hack for my dataset
  H = np.genfromtxt(os.path.join(data_path, 'H.txt'),
                    delimiter='  ',
                    unpack=True).transpose()

  # Extract static obstacles
  obst_threshold = 200
  static_obst_img = cv.imread(os.path.join(data_path, file_name), 0)
  #static_obst_img = cv.imread(os.path.join(load_data_path, 'map.png'), 0)
  obstacles = np.zeros([0, 3])
  #grid = (static_obst_img/255*-1)+1
  #static_obst_img = np.transpose(static_obst_img)
  for xx in range(static_obst_img.shape[0]):
    for yy in range(static_obst_img.shape[1]):
      if static_obst_img[xx, yy] > obst_threshold:
        obstacles = np.append(obstacles,
                              np.dot(H, np.array([[xx], [yy], [1]])).transpose(),
                              axis=0)
  Hinv = np.linalg.inv(H)

  # Compute obstacles in 2D
  obstacles_2d = np.zeros([obstacles.shape[0], 2])
  obstacles_2d[:, 0] = obstacles[:, 0] / obstacles[:, 2]
  obstacles_2d[:, 1] = obstacles[:, 1] / obstacles[:, 2]

  # Get obstacle idx on map
  obst_idx = []
  for obst_ii in range(obstacles_2d.shape[0]):
    idx = np.dot(Hinv, np.array([[obstacles_2d[obst_ii, 0]], [obstacles_2d[obst_ii, 1]], [1]])).transpose()
    obst_idx.append((int(idx[0,0]),int(idx[0,1])))
    grid_map['Map'][obst_idx[-1]] = 1


  grid_map['Closest X'] = np.zeros_like(grid_map['Map']) + 10000
  grid_map['Closest Y'] = np.zeros_like(grid_map['Map']) + 10000
  """ """
  # Calculate closest obstacle
  obst_idx = np.array(obst_idx)
  for xx in range(int(grid_map['Size'][0] / grid_map['Resolution'])):
    for yy in range(int(grid_map['Size'][1] / grid_map['Resolution'])):
      delta_idx = obst_idx - np.array([xx, yy])
      distances = np.sqrt(np.sum(np.square(delta_idx), axis=1))
      closest_obj = np.argmin(distances)
      grid_map['Closest X'][xx, yy] = obst_idx[closest_obj][0]
      grid_map['Closest Y'][xx, yy] = obst_idx[closest_obj][1]

  grid_map['Closest X'] = grid_map['Closest X'] * grid_map['Resolution'] - map_center[0]
  grid_map['Closest Y'] = grid_map['Closest Y'] * grid_map['Resolution'] - map_center[1]

  np.save(os.path.join(data_path, 'map'), grid_map)

def idx_from_pos(x, y, center, res=0.1):
  idx_x = round((x + float(center[0])) / res)
  idx_y = round((y + float(center[1])) / res)

  # Projecting index on map if out of bounds
  idx_x = max(0, min(idx_x, -1 + center[0] * 2. / res))
  idx_y = max(0, min(idx_y, -1 + center[1] * 2. / res))
  return int(idx_x), int(idx_y)

def path_from_vel(initial_pos, pred_vel,v0=np.array([0,0]), dt=0.1, n_vx=1, n_vy=1):
  """
  Extract a path from vector of predicted velocities by applying Euler forward integration.
  """
  vel_x = pred_vel[:,0]*n_vx
  vel_y = pred_vel[:,1]*n_vy
  n_steps = len(vel_x)
  pos_vec = np.zeros([n_steps, 2])
  pos_vec[0, :] = initial_pos + v0*dt
  for ii in range(1, n_steps):
    pos_vec[ii, :] = pos_vec[ii-1,:] + dt * np.array([vel_x[ii], vel_y[ii]])

  return pos_vec

def to_image_frame(Hinv, loc):
  """
	Given H^-1 and world coordinates, returns (u, v) in image coordinates.
	"""
  locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
  if locHomogenous.ndim > 1:
    loc_tr = np.transpose(locHomogenous)
    loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
    locXYZ = np.transpose(loc_tr / loc_tr[2])  # to pixels (from millimeters)
    return locXYZ[:, :2].astype(int)
  else:
    locHomogenous = np.dot(Hinv, locHomogenous)  # to camera frame
    locXYZ = locHomogenous / locHomogenous[2]  # to pixels (from millimeters)
    return locXYZ[:2].astype(int)

def to_pos_frame(Hinv, loc):
  """
	Given H^-1 and world coordinates, returns (u, v) in image coordinates.
	"""
  locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
  if locHomogenous.ndim > 1:
    loc_tr = np.transpose(locHomogenous)
    loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
    locXYZ = np.transpose(loc_tr / loc_tr[2])  # to pixels (from millimeters)
    return locXYZ[:, :2]
  else:
    locHomogenous = np.dot(Hinv, locHomogenous)  # to camera frame
    locXYZ = locHomogenous / locHomogenous[2]  # to pixels (from millimeters)
    return locXYZ[:2]

def line_cv(im, ll, value, width):
  for tt in range(ll.shape[0] - 1):
    cv2.line(im, (int(ll[tt][1]), int(ll[tt][0])), (int(ll[tt + 1][1]), int(ll[tt + 1][0])), value, width)

def plotGrid(grid, ax, color='k', alpha=1.0):
  plot_values_ped = np.zeros([0, 2])
  plot_values_grid = np.zeros([0, 2])
  for idx_x in range(grid.shape[0]):
    for idx_y in range(grid.shape[1]):
      if grid[idx_x, idx_y] > 0.8 and grid[idx_x, idx_y] < 1.5:
        plot_values_grid = np.append(plot_values_grid, np.array([[idx_x, idx_y]]), axis=0)
      elif grid[idx_x, idx_y] > 1.5:
        plot_values_ped = np.append(plot_values_ped, np.array([[idx_x, idx_y]]), axis=0)
  ax.plot(plot_values_grid[:, 0], plot_values_grid[:, 1], marker='o', color=color, lw=0)
  ax.plot(plot_values_ped[:, 0], plot_values_ped[:, 1], marker='o', color='r', lw=0)
  ax.set_xlim([0, grid.shape[0]])
  ax.set_ylim([0, grid.shape[1]])