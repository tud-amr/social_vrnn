import numpy as np
from scipy import interpolate

class Trajectory():
  
  def __init__(self, time_vec = np.zeros([0]), pose_vec = np.zeros([0,3]), vel_vec = np.zeros([0,3]), goal = np.zeros([1,2])):
    self.time_vec = time_vec    # timesteps in [ns]
    self.pose_vec = pose_vec    # [x, y, heading]
    self.vel_vec = vel_vec      # [vx, vy, omega]
    self.goal = goal            # [x,y]
    self.pose_interp = None
    self.vel_interp = None
    self.other_agents_positions = []  # store indices of other agents trajectories for each time step of this trajectory (with wich other positions does it need to be compared at a certain time)
    self.other_agents_velocities = []
    
  def __len__(self):
    if len(self.time_vec) == len(self.pose_vec) == len(self.vel_vec):
      return len(self.time_vec)
    else: 
      raise Exception('Member vectors are of different length. Cannot get unique length.')
    
  def addData(self, timestamp, pose, vel, goal):
    """
    Inserts the provided data at the right index in the trajectory.
    """
    assert np.all(goal == self.goal), "Goals have to be matching."
    # Add data at the right position. In the log the data might not be ordered properly.
    if self.time_vec.shape[0] > 0 and timestamp < self.time_vec[-1]:
      try: 
        idx_insert = np.where(timestamp > self.time_vec)[-1][-1] + 1
      except:
        idx_insert = 0  # In case the timestamp is smaller than the first element of time_vec
      self.time_vec = np.insert(self.time_vec, idx_insert, timestamp)
      self.pose_vec = np.insert(self.pose_vec, idx_insert, np.atleast_2d(pose), axis=0)
      self.vel_vec = np.insert(self.vel_vec, idx_insert, np.atleast_2d(vel), axis=0)
    else:
      self.time_vec = np.append(self.time_vec, timestamp)
      self.pose_vec = np.append(self.pose_vec, np.atleast_2d(pose), axis=0)
      self.vel_vec = np.append(self.vel_vec, np.atleast_2d(vel), axis=0)
    
    if self.time_vec.shape[0] > 1:
      self.updateInterpolators()
      
  def subsample(self, subsampling_factor):
    """
    Subsample the trajectory in order to reduce the overall number of samples and 
    increase the time difference between two samples.
    """
    if int(self.time_vec[0]*10) % 2 == 1:
      self.time_vec = self.time_vec[1:]
      self.pose_vec = self.pose_vec[1:, :]
      self.vel_vec = self.vel_vec[1:, :]
    self.time_vec = self.time_vec[0::subsampling_factor]
    self.pose_vec = self.pose_vec[0::subsampling_factor, :]
    self.vel_vec = self.vel_vec[0::subsampling_factor, :]

    
    #self.updateInterpolators()
        
  def updateInterpolators(self):
    self.pose_interp = interpolate.interp1d(self.time_vec, self.pose_vec, kind='linear', axis=0)  # TODO: correct heading interpolation
    self.vel_interp = interpolate.interp1d(self.time_vec, self.vel_vec, kind='linear', axis=0)
      
  def contains(self, query_time):
    return query_time >= self.time_vec[0] and query_time <= self.time_vec[-1]
   
  def getPoseAtTime(self, query_time):
    pose = self.pose_interp(query_time)
    return pose
   
  def getVelAtTime(self, query_time):
    return self.vel_interp(query_time)
  
  def getDataAtTime(self, query_time):
    pose_interpol = self.getPoseAtTime(query_time)
    vel_interpol = self.getVelAtTime(query_time)
    return pose_interpol, vel_interpol
  
  def getMinTime(self):
    """
    Return the starting time of the trajectory.
    """
    return self.time_vec[0]
  
  def getMaxTime(self):
    """
    Return the end time of the trajectory.
    """
    return self.time_vec[-1]  
  
  def getDuration(self):
    return self.getMaxTime() - self.getMinTime()
  
  def smoothenTrajectory(self, dt=0.3):
    """
    Cubic interpolation with provided values in order to smoothen the trajectory and obtain
    a sequence with the specified dt.
    """
    x_interpolator = interpolate.interp1d(self.time_vec, self.pose_vec[:,0], kind='cubic', axis=0)
    y_interpolator = interpolate.interp1d(self.time_vec, self.pose_vec[:,1], kind='cubic', axis=0)
    vx_interpolator = interpolate.interp1d(self.time_vec, self.vel_vec[:,0], kind='cubic', axis=0)
    vy_interpolator = interpolate.interp1d(self.time_vec, self.vel_vec[:,1], kind='cubic', axis=0)
    n_elem = int((self.getMaxTime() - self.getMinTime()) / dt)
    new_time_vec = np.linspace(self.getMinTime(), self.getMinTime() + (n_elem - 1) * dt, n_elem)
    
    new_time_vec = np.zeros([n_elem])
    new_pose_vec = np.zeros([n_elem, 3])
    new_vel_vec = np.zeros([n_elem, 3])
    
    for ii in range(n_elem):
      t = self.getMinTime() + ii*dt
      new_time_vec[ii] = t
      new_pose_vec[ii, 0] = x_interpolator(t)
      new_pose_vec[ii, 1] = y_interpolator(t)
      new_vel_vec[ii, 0] = vx_interpolator(t)
      new_vel_vec[ii, 1] = vy_interpolator(t)
    
    self.goal = new_pose_vec[-1,:2]
    self.time_vec = new_time_vec
    self.pose_vec = new_pose_vec
    self.vel_vec = new_vel_vec
    

  def __getstate__(self):
    d = dict(self.__dict__)
    del d['pose_interp']
    del d['vel_interp']
    return d
    