import numpy as np
import sys
if sys.version_info[0] < 3:
  import Trajectory as traj
else:
  import src.data_utils.Trajectory as traj

class AgentData():
  """
  Contains data for one agent, comprising all trajectories the agent traveled within the given map.
  Different trajectories are separated by their different goals.
  """
  
  def __init__(self, id, radius=0.25):
    self.id = id
    self.goals = []
    self.trajectories = []
    self.radius = radius
    self.cached_trajectory_idx = 0
    self.traj_idx = 0
    self.last_goal = 0
    
  def addSample(self, timestamp, pose, vel, goal):
    """
    Automatically appends the input state to the matching trajectory. 
    If this measurement is part of a new trajectory, one will be created.
    """
    # hack to catch situation of changing motion direction
    #if goal[0] < 0:
    #  goal = self.last_goal
    if goal not in self.goals:

        # adding goal identifier
        self.goals.append(goal)
        self.trajectories.append(traj.Trajectory(goal=np.array([[goal[0], goal[1]]])))
        if len(self.goals) > 1:
          self.traj_idx += 1
    else:
      if len(self.goals)>0:
        if not np.all(goal == self.last_goal):
          self.trajectories.append(traj.Trajectory(goal=np.array([[goal[0], goal[1]]])))
          self.traj_idx += 1

    # append data sample to trajectory (timestamp in [ns], pose and vel as np 1x3 arrays)
    self.trajectories[self.traj_idx].addData(timestamp, pose, vel, goal)
    self.last_goal = goal
    
  
  def getTrajectoryForTime(self, query_time):
    """
    Return the matching trajectory for the query time.
    """
    if self.trajectories[self.cached_trajectory_idx].contains(query_time):
      return self.trajectories[self.cached_trajectory_idx]
    elif query_time >= self.trajectories[0].getMinTime() and query_time <= self.trajectories[-1].getMaxTime():
      # Find trajectory that contains the query time
      for idx, t in enumerate(self.trajectories):
        if t.contains(query_time):
          self.cached_trajectory_idx = idx
          return t
    else:
      return None
  
  def plot(self, ax, color='b', x_scale = 1, y_scale = 1):
    for t in self.trajectories:
      ax.plot(t.pose_vec[:,0]*x_scale, t.pose_vec[:,1]*y_scale, color=color)
