import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.insert(0, 'MapUtils')
from bresenham2D import bresenham2D
import load_data as ld

import time 
import math 
import transformations as tf 
from math import cos, sin 
class JOINTS:
	"""Return data collected from IMU and anything not related to lidar
	return 
	self.data_['ts'][0]: 1 x N array of absolute time values 
	self.data_['pos']: 35xN array of sth we donnot care about 
	self.data_['rpy']: 3x N array of roll, pitch, yaw angles over time 
	self.data_['head_angles']: 2x N array of head angles (neck angle, head angle)
	"""
	def __init__(self,dataset='0',data_folder='data',name=None):
		if name == None:
			joint_file = os.path.join(data_folder,'train_joint'+dataset)
		else:
			joint_file = os.path.join(data_folder,name)
		joint_data = ld.get_joint(joint_file)
		self.num_measures_ = len(joint_data['ts'][0])
		self.data_ = joint_data 
		self.head_angles = self.data_['head_angles'] 

class LIDAR:
	def __init__(self,dataset='0',data_folder='data',name=None):
		if name == None:
			lidar_file = os.path.join(data_folder,'train_lidar' +dataset)
		else:
			lidar_file = os.path.join(data_folder,name)
		lidar_data = ld.get_lidar(lidar_file)
		# substract the beginning value of yaw given by ypr
		# lidar_data[:]['rpy'][0]
		yaw_offset = lidar_data[0]['rpy'][0,2]
		position_offset = lidar_data[0]['pose'][0]
		
		for j in range(len(lidar_data)):
			lidar_data[j]['rpy'][0,2] -= yaw_offset
			lidar_data[j]['pose'][0] -= position_offset
		#self.range_theta_ = (np.arange(0,270.25,0.25) - 135)*np.pi/float(180)
		self.num_measures_ = len(lidar_data)
		self.data_ = lidar_data
		# self._read_lidar(lidar_data)
		# Limitation of the lidar's ray 
		#self.L_MIN = 0.001
		#self.L_MAX = 30
		#self.res_ = 0.25 # (angular resolution of rada = 0.25 degrees)

	#def ray_in_world(self,R_pose,head_angle, neck_angle,lidar_rays, valid_range):
	def ray_in_world(self,R_pose,head_angle, neck_angle,lidar_rays, theta):
            homo_lidar_to_head = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.15],[0,0,0,1]])
            homo_head_to_body = tf.homo_transform(np.dot(tf.rot_z_axis(neck_angle), tf.rot_y_axis(head_angle)),np.array([0,0,0.33]))
            homo_body_to_ground = tf.homo_transform(tf.rot_z_axis(R_pose[2]) , np.array([R_pose[0],R_pose[1],0.93]))

            homo_lidar_to_body = np.dot(homo_head_to_body , homo_lidar_to_head)
            homo_lidar_to_ground = np.dot(homo_body_to_ground,homo_lidar_to_body)
            #ray_angle = theta.reshape((1,theta.shape[0]))

            lidar_rays_x = lidar_rays * np.cos(theta)#(ray_angle)
            lidar_rays_y = lidar_rays * np.sin(theta)#(ray_angle)
            #lidar_rays_z = np.zeros((1,lidar_rays_x.shape[1]))
            #ones = np.ones((1,lidar_rays_x.shape[1]))
            ray_ground = np.dot( homo_lidar_to_ground , np.array( [lidar_rays_x, lidar_rays_y, np.zeros(lidar_rays_x.shape[0]), np.ones(lidar_rays_x.shape[0])] )) #np.vstack( (lidar_rays_x, lidar_rays_y, lidar_rays_z, ones) ) )
            return ray_ground[:,ray_ground[2,:] > 0.1]
		
	def map_indices(self,MAP,positionx , positiony):
		""" Return the corresponding indices in MAP array, given the physical position"""
		# convert from meters to cells

		xis = np.ceil((positionx - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
		yis = np.ceil((positiony - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
		#xis = np.ceil((MAP['ymax'] - positionx) / MAP['res'] ).astype(np.int16)-1
		#yis = np.ceil((MAP['ymax'] - positiony) / MAP['res'] ).astype(np.int16)-1
		return [xis, yis]