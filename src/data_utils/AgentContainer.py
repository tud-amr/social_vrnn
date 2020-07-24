import numpy as np
import pylab as pl
import sys
if sys.version_info[0] < 3:
  import Trajectory as traj
  import OccupancyGrid as occ
  import AgentData as ad
  import Support as sup
else:
  import src.data_utils.Trajectory as traj
  import src.data_utils.OccupancyGrid as occ
  import src.data_utils.AgentData as ad
  import src.data_utils.Support as sup


class AgentContainer():
  """
  Container for all agents in one demonstration. 
  This data structure provides an interface to easily store and access data of various agents.
  It contains a dictionary of the agent data per agent id and the static occupancy grid.
  """
  def __init__(self):
    self.agent_data = {}
    self.occupancy_grid = occ.OccupancyGrid()
    
  def setOccupancyGrid(self, occupancy_grid):
    self.occupancy_grid = occupancy_grid
  
  def removeAgent(self, id):
    self.agent_data.pop(id)
  
  def addDataSample(self, id, timestamp, pose, vel, goal):
    """
    Add a data sample to the database consisting of agent data per ID.
    """
    # set up new agent if id has not been registered yet
    if id not in self.agent_data.keys():
      self.agent_data[id] = ad.AgentData(id=id)
    
    # add the data to the agent with the right id
    self.agent_data[id].addSample(timestamp, pose, vel, tuple(goal))
  
  
  # Getters and setters
  def getAgentIDs(self):
    """
    Return list of IDs of contained agents.
    """
    return [int(k) for k in self.agent_data.keys()]
  
  def getNumberOfAgents(self):
    """
    Get the number of agents in the container. 
    """
    return len(self.agent_data.keys())
  
  def getAgentTrajectories(self, id):
    """
    Get all trajectories of a certain agent.
    """
    return self.agent_data[id].trajectories
  
  def getAgentData(self, id):
    """
    Return the agent data of agent with ID=id
    """
    return self.agent_data[id]
  
  def getNumberOfTrajectoriesForAgent(self, id):
    """
    Return the number of trajectories (start -> goal) of agent with ID=id.
    """
    return len(self.agent_data[id].trajectories)
  
  def getTrajectorySetForTime(self, query_time):
    """
    Get all trajectories which are available at a certain time.
    """
    traj_set = {}
    for id in self.agent_data.keys():
      t = self.agent_data[id].getTrajectoryForTime(query_time)
      if t != None:
        traj_set[id] = t
    return traj_set
    
  def getAgentPositionsForTimeExclude(self, query_time, agent_id):
    """
    Get all agent positions for certain time.
    query_time: time where the entries should be queried
    agent_id: id to exclude
    """
    pos_array = np.zeros([0, 2])
    cnt = 0
    for id in self.agent_data.keys():
      if id != agent_id:
        t = self.agent_data[id].getTrajectoryForTime(query_time)
        if t != None and len(t) > 1:
          pos_array = np.append(pos_array, np.expand_dims(t.getPoseAtTime(query_time)[0:2], axis=0), axis=0)
          cnt += 1
    return pos_array
  
  def getAgentVelocitiesForTimeExclude(self, query_time, agent_id):
    """
    Get all agent positions for certain time.
    query_time: time where the entries should be queried
    agent_id: id to exclude
    """
    vel_array = np.zeros([0, 2])
    cnt = 0
    for id in self.agent_data.keys():
      if id != agent_id:
        t = self.agent_data[id].getTrajectoryForTime(query_time)
        if t != None and len(t) > 1:
          vel_array = np.append(vel_array, np.expand_dims(t.getVelAtTime(query_time)[0:2], axis=0), axis=0)
          cnt += 1
    return vel_array
  
  def plot(self, ax, x_scale = 1, y_scale = 1):
    """
    Plots the trajectories and the static occupancy grid of all agents in this container.
    """
    colormap = pl.get_cmap('rainbow')
    c_norm = pl.matplotlib.colors.Normalize(vmin=0, vmax=len(self.agent_data.keys()))
    scalar_color_map = pl.cm.ScalarMappable(norm=c_norm, cmap=colormap)
    
    for cnt, id in enumerate(self.agent_data.keys()):
      color_value = scalar_color_map.to_rgba(cnt)
      self.agent_data[id].plot(ax, color=color_value,x_scale = 1, y_scale = 1)
      
    #sup.plot_grid(ax, np.array([self.occupancy_grid.center[0], self.occupancy_grid.center[1]]), self.occupancy_grid.gridmap, self.occupancy_grid.resolution, self.occupancy_grid.map_size)
    sup.plot_grid(ax, np.array([0, 0]),self.occupancy_grid.gridmap, self.occupancy_grid.resolution, self.occupancy_grid.map_size)
    ax.set_xlim([-self.occupancy_grid.center[0], self.occupancy_grid.center[0]])
    ax.set_ylim([-self.occupancy_grid.center[1], self.occupancy_grid.center[1]])
    
    ax.set_aspect('equal')
    
  