#*
#    SLAM.py: the implementation of SLAM
#    created and maintained by Ty Nguyen
#    tynguyen@seas.upenn.edu
#    Feb 2020
#*
import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import os, sys, time
import p3_util as ut
from read_data import LIDAR, JOINTS
import probs_utils as prob
import math
import cv2
import transformations as tf
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import logging
if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle

import tqdm
from bresenham2D import bresenham2D

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
 

class SLAM(object):
    def __init__(self, p_thresh = 0.6):
        self.log_p_true_ = math.log(9)
        self.log_p_false_ = math.log(1.0/9)
        self.logodd_thresh_ = prob.log_thresh_from_pdf_thresh(p_thresh)
        self.range_theta_ = (np.arange(0,270.25,0.25) - 135)*np.pi/float(180)
    
    def _read_data(self, src_dir, dataset=0, split_name='train'):
        self.dataset_= str(dataset)
        if split_name.lower() not in src_dir:
            src_dir  = src_dir + '/' + split_name
        print('\n------Reading Lidar and Joints (IMU)------')
        self.lidar_  = LIDAR(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_lidar'+ self.dataset_)
        print ('\n------Reading Joints Data------')
        self.joints_ = JOINTS(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_joint'+ self.dataset_)

        self.num_data_ = len(self.lidar_.data_)
        # Position of odometry
        #self.odo_indices_ = np.empty((2,self.num_data_),dtype=np.int64)
        

    def _init_particles(self, num_p=0, mov_cov=None, particles=None, weights=None, percent_eff_p_thresh=None):
        # Particles representation
        self.num_p_ = num_p
        #self.percent_eff_p_thresh_ = percent_eff_p_thresh
        self.particles_ = np.zeros((3,self.num_p_),dtype=np.float64) if particles is None else particles
        
        # Weights for particles
        self.weights_ = 1.0/self.num_p_*np.ones(self.num_p_) if weights is None else weights

        # Position of the best particle after update on the map
        #self.best_p_indices_ = np.empty((2,self.num_data_),dtype=np.int64)
        self.best_p_indices_ = np.zeros((2,self.num_data_)).astype(np.int16)
        # Best particles
        self.best_p_ = np.empty((3,self.num_data_))
        # Corresponding time stamps of best particles
        #self.time_ =  np.empty(self.num_data_)
       
        # Covariance matrix of the movement model
        tiny_mov_cov   = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0 , 1e-8]])
        self.mov_cov_  = mov_cov if mov_cov is not None else tiny_mov_cov
        # To generate random noise: x, y, z = np.random.multivariate_normal(np.zeros(3), mov_cov, 1).T
        # this return [x], [y], [z]

        # Threshold for resampling the particles
        self.percent_eff_p_thresh_ = percent_eff_p_thresh

    def _init_map(self, map_resolution=0.05):
        '''*Input: resolution of the map - distance between two grid cells (meters)'''
        # Map representation
        MAP= {}
        MAP['res']   = map_resolution #meters
        MAP['xmin']  = -20  #meters
        MAP['ymin']  = -20
        MAP['xmax']  =  20
        MAP['ymax']  =  20
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        self.MAP_ = MAP

        self.log_odds_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        self.occu_ = np.ones((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.uint64)


    def _build_first_map(self,t0=0,use_lidar_yaw=True):
        """Build the first map using first lidar"""
        self.t0 = t0
        # Extract a ray from lidar data
        MAP = self.MAP_
        print('\n--------Doing build the first map--------')

        #TODO: student's input from here 
        # Get index of closest timestamp in joint data to retrieve head angles
        lidar_time = self.lidar_.data_[self.t0]['t'][0][0]
        time_diff = abs(self.joints_.data_['ts'][0] - lidar_time)
        joint_index = np.where(time_diff == min(time_diff) )[0][0]
        angles = self.joints_.data_['head_angles'][:,joint_index]

        pose = self.lidar_.data_[self.t0]['pose'][0]
        #pose = np.array([0,0,0])
        #pose[2] = -1.03816304
        #get lidar_scan
        lidar_scan = self.lidar_.data_[self.t0]['scan'][0]
        valid_range = np.logical_and(lidar_scan > 0.1 , lidar_scan <30)
        valid_lidar_scan = lidar_scan[valid_range]
        theta = self.range_theta_[valid_range]
        
        #[ray_world_start , ray_world_end] = 
        #ray_world = self.lidar_.ray_in_world(R_pose = pose , head_angle = angles[1], neck_angle = angles[0] , lidar_rays = lidar_scan[valid_range],valid_range = valid_range)
        #ray_ground = self.lidar_._remove_ground(ray_world=ray_world)
        #ray_ground = self.lidar_.ray_in_world(R_pose = pose , head_angle = angles[1], neck_angle = angles[0] , lidar_rays = lidar_scan[valid_range],valid_range = valid_range)
        ray_ground = self.lidar_.ray_in_world(R_pose = pose , head_angle = angles[1], neck_angle = angles[0] , lidar_rays = valid_lidar_scan,theta = theta)

        [ray_end_x , ray_end_y] = self.lidar_.map_indices(MAP , positionx = ray_ground[0,:] , positiony = ray_ground[1,:])
        [ray_start_x , ray_start_y] = self.lidar_.map_indices(MAP , positionx = pose[0] , positiony = pose[1])
        ray_start_x = np.tile(ray_start_x ,ray_end_x.shape[0])
        ray_start_y = np.tile(ray_start_y ,ray_end_x.shape[0])
        map_indices = np.vstack((ray_start_x,ray_start_y,ray_end_x,ray_end_y))
        
        #map_indices_occupied = np.unique(map_indices_occupied , axis = 1)
        #for j in map_indices_occupied.T:
        for j in map_indices.T:
            #cells = bresenham2D(j[0] , j[1] , j[2] , j[3])
            #cells = cells.astype(np.int16)
            #self.log_odds_[cells[0,-1], cells[1,-1]] += self.log_p_true_
            #self.log_odds_[cells[0,:-1] , cells[1,:-1]] += self.log_p_false_
            self.log_odds_[j[2], j[3]] += self.log_p_true_
            self.log_odds_[j[0], j[1]] += self.log_p_false_

        """    
        for j in range(ray_world.shape[1]):
            cells = self.lidar_._cellsFrom2Points([sX_map[j],sY_map[j],eX_map[j],eY_map[j]])
            cells = cells.astype(np.int16)
            #if np.any(cells > 800):
                #continue
            if ray_remove_ground[2,j]:
                self.log_odds_[cells[0,-1], cells[1,-1]] += self.log_p_true_
                self.log_odds_[cells[0,:-1], cells[1,:-1]] += self.log_p_false_
                #self.log_odds_[cells[1,-1] , cells[0,-1]] += self.log_p_true_
                #self.log_odds_[cells[1,:-1] , cells[0,:-1]] += self.log_p_false_
            else:
                self.log_odds_[cells[0,:] , cells[1,:]] += self.log_p_false_
                #self.log_odds_[cells[1,:] , cells[0,:]] += self.log_p_false_
        """
        self.best_p_[:,0] = [0,0,0]
        self.best_p_indices_[:,0] = [400,400]
        #End student's input
        MAP['map'] = self.log_odds_

        self.MAP_ = MAP



    def _predict(self,t,use_lidar_yaw=True):
        #logging.debug('\n-------- Doing prediction at t = {0}------'.format(t))
        #TODO: student's input from here 
        x_tminus1 = self.lidar_.data_[t-1]['pose'][0]
        x_tminus1[2] = self.lidar_.data_[t-1]['rpy'][0,2]
        
        x_t = self.lidar_.data_[t]['pose'][0]
        x_t[2] = self.lidar_.data_[t]['rpy'][0,2]

        #relative_movement = tf.twoDSmartMinus(self.lidar_.data_[t]['pose'][0] , self.lidar_.data_[t-1]['pose'][0])
        relative_movement = tf.twoDSmartMinus(x_t , x_tminus1)
        for i in range(0,self.num_p_):
            self.particles_[:,i] = tf.twoDSmartPlus(self.particles_[:,i] , relative_movement)
            temp = np.random.multivariate_normal(np.array([0,0,0]) , self.mov_cov_)
            self.particles_[:,i] = tf.twoDSmartPlus(self.particles_[:,i] , temp)
            #self.particles_[0:2,i] += np.random.multivariate_normal(np.array([0,0,0]) , self.mov_cov_)[0:2]
        #self.particles_ = np.random.multivariate_normal(np.array([0,0,0]), self.mov_cov_, size = (10)).T
        #self.particles_[2,:] = self.lidar_.data_[t]['rpy'][0,2]
        #End student's input 

    def _update(self,t):
        """Update function where we update the """
        #if t == t0:
            #self._build_first_map(t0,use_lidar_yaw=True)
            #return

        #TODO: student's input from here 
        MAP = self.MAP_

        lidar_time = self.lidar_.data_[t]['t'][0][0]
        time_diff = abs(self.joints_.data_['ts'][0] - lidar_time)
        joint_index = np.where(time_diff == min(time_diff) )[0][0]
        angles = self.joints_.data_['head_angles'][:,joint_index]
        

        lidar_scan = self.lidar_.data_[t]['scan'][0]
        correlations = np.zeros(self.weights_.shape[0])

        valid_range = np.logical_and(lidar_scan > 0.1 , lidar_scan <30)
        valid_lidar_scan = lidar_scan[valid_range]
        theta = self.range_theta_[valid_range]
        head_angle = angles[1]
        neck_angle = angles[0]
        #lidar_pose = self.lidar_.data_[t]['pose'][0]

        binary_map = MAP['map'] > self.logodd_thresh_#Compute correlations
        for i in range(0,self.num_p_):
            #pose = self.particles_[:,i] + lidar_pose
            pose = self.particles_[:,i]
            #ray_world = self.lidar_.ray_in_world(R_pose = pose , head_angle = angles[1], neck_angle = angles[0] , lidar_rays = lidar_scan[valid_range],valid_range = valid_range)
            #ray_ground = self.lidar_._remove_ground(ray_world=ray_world)
            #ray_ground = self.lidar_.ray_in_world(R_pose = pose , head_angle = angles[1], neck_angle = angles[0] , lidar_rays = lidar_scan[valid_range],valid_range = valid_range)
            ray_ground = self.lidar_.ray_in_world(R_pose = pose , head_angle = head_angle, neck_angle = neck_angle , lidar_rays = valid_lidar_scan,theta = theta)

            #[ray_end_x , ray_end_y] = self.lidar_.map_indices(MAP , positionx = ray_ground[0,:] , positiony = ray_ground[1,:])

            #corr_indices = np.vstack((ray_end_x, ray_end_y))
            #corr_indices = corr_indices.astype(np.int16)
            corr_indices = np.ceil((ray_ground + 20)/0.05).astype(np.int16) - 1

            if np.any(corr_indices >800):
                continue
            #correlations[i] = prob.mapCorrelation( MAP['map'] , corr_indices)
            correlations[i] = np.sum(binary_map[corr_indices[0,:],corr_indices[1,:]])#prob.mapCorrelation( ( MAP['map'] > self.logodd_thresh_ ), corr_indices)

        self.weights_ = prob.update_weights(self.weights_ , correlations)
        max_weight_index = np.argmax(self.weights_)
        self.best_p_[:,t] = self.particles_[:,max_weight_index]
        [x_best , y_best] = self.lidar_.map_indices(MAP, positionx = self.best_p_[0,t], positiony = self.best_p_[1,t])
        
        self.best_p_indices_[:,t] = [x_best, y_best]

        #update the map according to this best particle
        pose = self.best_p_[:,t]

        #ray_world = self.lidar_.ray_in_world(R_pose = pose , head_angle = angles[1], neck_angle = angles[0] , lidar_rays = lidar_scan[valid_range],valid_range = valid_range)
        #ray_ground = self.lidar_._remove_ground(ray_world=ray_world)
        #ray_ground = self.lidar_.ray_in_world(R_pose = pose , head_angle = angles[1], neck_angle = angles[0] , lidar_rays = lidar_scan[valid_range],valid_range = valid_range)
        ray_ground = self.lidar_.ray_in_world(R_pose = pose , head_angle = angles[1], neck_angle = angles[0] , lidar_rays = valid_lidar_scan,theta = theta)

        [ray_end_x , ray_end_y] = self.lidar_.map_indices(MAP , positionx = ray_ground[0,:] , positiony = ray_ground[1,:])
        [ray_start_x , ray_start_y] = self.lidar_.map_indices(MAP , positionx = pose[0] , positiony = pose[1])
        ray_start_x = np.tile(ray_start_x ,ray_end_x.shape[0])
        ray_start_y = np.tile(ray_start_y ,ray_end_x.shape[0])
        map_indices = np.vstack((ray_start_x,ray_start_y,ray_end_x,ray_end_y))
        #map_indices = np.unique(map_indices, axis = 1)
        #self.log_odds_[ray_end_x, ray_end_y] += self.log_p_true_#map_indices = np.unique(map_indices, axis = 1)

        #for j in range(ray_end_x.shape[0]):#
        for j in map_indices.T:
            #cells = bresenham2D(j[0] , j[1] , j[2] , j[3]).astype(np.int)
            #if np.any(cells > 800):
                #continue
            #self.log_odds_[cells[0,-1], cells[1,-1]] += self.log_p_true_
            #self.log_odds_[cells[0,:-1] , cells[1,:-1]] += self.log_p_false_
            if np.any(j > 800):
                continue
            self.log_odds_[j[2], j[3]] += self.log_p_true_
            self.log_odds_[j[0] , j[1]] += self.log_p_false_

        
        MAP['map'] = self.log_odds_

        #End student's input 

        self.MAP_ = MAP
        return MAP
