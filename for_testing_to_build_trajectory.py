import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import os, sys, time
import p3_util as ut
from read_data import LIDAR, JOINTS
import probs_utils as prob
import math
import transformations as tf
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from SLAM import SLAM
import argparse
import pdb 
import tqdm 
from gen_figures import genMap
import cv2
import logging

def particle_SLAM(src_dir, dataset_id=0, split_name='train', running_mode='test_SLAM', log_dir='logs'):
    '''Your main code is here.
    '''
    ###############################################################################################  
    #* Student's input
    #TODO: change the resolution of the map - the distance between two cells (meters)
    map_resolution = 0.05    

    # Number of particles 
    #TODO: change the number of particles
    num_p = 100

    #TODO: change the process' covariance matrix 
    mov_cov = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0 , 1e-8]])
        
    #TODO: set a threshold value of probability to consider a map's cell occupied  
    p_thresh = 0.6 
    
    #TODO: change the threshold of the percentage of effective particles to decide resampling 
    percent_eff_p_thresh = 0.5
    
    #*End student's input
    ###############################################################################################  
    """    
    # Test prediction
    if running_mode == 'test_prediction':
        test_prediction(src_dir, dataset_id, split_name, log_dir)
        exit(1)
    if running_mode == 'test_update':
        test_update(src_dir, dataset_id, split_name, log_dir, map_resolution)
        exit(1)
    """
    # Test SLAM
    # Create a SLAM instance
    slam_inc = SLAM()
    
    # Read data
    slam_inc._read_data(src_dir, dataset_id, split_name)
    num_steps = slam_inc.num_data_
    
    # Characterize the sensors' specifications
    slam_inc._characterize_sensor_specs(p_thresh)

    # Initialize particles
    slam_inc._init_particles(num_p, mov_cov, percent_eff_p_thresh=percent_eff_p_thresh) 

    # Iniitialize the map
    slam_inc._init_map(map_resolution)
   
    # Starting time index
    t0 = 0 

    # Initialize the particle's poses using the lidar measurements at the starting time
    slam_inc.particles_[:,0] = slam_inc.lidar_.data_[t0]['pose'][0]
    # Indicator to notice that map has not been built
    build_first_map = False

    # iterate next time steps
    all_particles = deepcopy(slam_inc.particles_)
    num_resamples = 0
    
    robot_pose = np.zeros((3,len(slam_inc.lidar_.data_)))
    robot_pose[:,0] = slam_inc.lidar_.data_[0]['pose'][0]

    for t in tqdm.tqdm(range(1,num_steps)): #(range(t0, num_steps - t0)):
        relative_movement = tf.twoDSmartMinus(slam_inc.lidar_.data_[t]['pose'][0] , slam_inc.lidar_.data_[t-1]['pose'][0])
        robot_pose[:,t] = tf.twoDSmartPlus(robot_pose[:,t-1] , relative_movement)
        
    #print(robot_pose[:,0])
    plt.figure(1)
    plt.plot(robot_pose[0,:] , robot_pose[1,:])
    plt.show()

    """
    for t in tqdm.tqdm(range(t0, num_steps-t0)):
        # Ignore lidar scans that are obtained before the first IMU
        if slam_inc.lidar_.data_[t]['t'][0][0] - slam_inc.joints_.data_['ts'][0][0] < 0:
            continue
        if not build_first_map:
            slam_inc._build_first_map(t)
            t0 = t
            build_first_map = True
            continue

        # Prediction
        slam_inc._predict(t)

        # Update
        slam_inc._update(t,t0=t0,fig='off')

        # Resample particles if necessary
        num_eff = 1.0/np.sum(np.dot(slam_inc.weights_,slam_inc.weights_))
        logging.debug('>> Number of effective particles: %.2f'%num_eff)
        if num_eff < slam_inc.percent_eff_p_thresh_*slam_inc.num_p_:
            num_resamples += 1
            logging.debug('>> Resampling since this < threshold={0}| Resampling times/t = {1}/{2}'.format(\
                slam_inc.percent_eff_p_thresh_*slam_inc.num_p_, num_resamples, t-t0 + 1))
            [slam_inc.particles_,slam_inc.weights_] = prob.stratified_resampling(\
                slam_inc.particles_,slam_inc.weights_,slam_inc.num_p_)
        
        # Plot the estimated trajectory
        if (t - t0 + 1)%1000 == 0 or t==num_steps-1:
            # Save the result 
            log_file = log_dir + '/SLAM_' + split_name + '_' + str(dataset_id) + '.pkl'
            try:
                with open(log_file, 'wb') as f:
                    pickle.dump(slam_inc,f,pickle.HIGHEST_PROTOCOL)
                print(">> Save the result to: %s"%log_file)
            except Exception as e:
                print('Unable to write data to', log_file, ':', e)
                raise

        
            # Gen map + trajectory
            MAP_2_display = genMap(slam_inc, t)
            MAP_fig_path = log_dir + '/processing_SLAM_map_'+ split_name + '_' + str(dataset_id) + '.jpg'
            cv2.imwrite(MAP_fig_path, MAP_2_display)
            plt.title('Estimated Map at time stamp %d/%d'%(t, num_steps - t0 + 1))
            plt.imshow(MAP_2_display)
            plt.pause(0.01)
        
            logging.debug(">> Save %s"%MAP_fig_path)
            
    # Return best_p which are an array of size 3xnum_data that represents the best particle over the whole time stamp
    return slam_inc.best_p_
    """

def main():
    parser = argparse.ArgumentParser('main function')
    parser.add_argument('--src_dir',    help="Directory to the data...i.e: data", default='data', type=str)
    parser.add_argument('--log_dir',    help="Directory to save logs",            default='logs', type=str)
    parser.add_argument('--dataset_id', help="Dataset id=0, 1, 2. ..?",           default=0, type=int)
    parser.add_argument('--split_name', help="Train or test split?",              default='train', type=str)
    parser.add_argument('--running_mode', help="Test prediction/Update/SLAM?",    default='test_SLAM', type=str,\
                                        choices=['test_SLAM', 'test_prediction', 'test_update'])

    args   = parser.parse_args()

    particle_SLAM(args.src_dir, args.dataset_id, args.split_name, args.running_mode, args.log_dir)

    # Run the particle SLAM 
    #best_p_array = particle_SLAM(args.src_dir, args.dataset_id, args.split_name, args.running_mode, args.log_dir)


if __name__ == "__main__":
    main()