#!/usr/bin/env python
# coding: utf-8
import numpy as np
import rospy
import subprocess
import message_filters
import sensor_msgs.point_cloud2 as pc2
import cv2
import sys
import os
import pickle
import torch 

from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2, CompressedImage
from nav_msgs.msg import Odometry
from pathlib import Path
import matplotlib.pyplot as plt
from transformer import ApplyTransformation
from model_builder.multimodal.fusion_net  import BcFusionModel


counter = {'index': 0, 'sub-sampler': 1}
previous_rbt_location = []
local_goal = {}
previous_velocities = []
play_back_snapshot = {}
image_history = []
pcl_history = []

device = "cuda" if torch.cuda.is_available() else "cpu"

def odom_callback(odom):
    position = odom.pose.pose.position
    cmd_vel = odom.twist.twist

    # print((cmd_vel.linear.x, cmd_vel.angular.z))
    # store 20 latest liner and angular velocities
    previous_velocities.insert(0, (cmd_vel.linear.x, cmd_vel.angular.z))
    if len(previous_velocities) > 20:
        previous_velocities.pop()

    return


def get_lidar_points(lidar):
    point_cloud = []
    # print("getting lidar...")
    for point in pc2.read_points(lidar, skip_nans=True, field_names=('x', 'y', 'z')):
        pt_x = point[0]
        pt_y = point[1]
        pt_z = point[2]
        point_cloud.append([pt_x, pt_y, pt_z])
    return point_cloud


def get_prev_cmd_val():
    return previous_velocities


def store_image(rgb_image):
    np_arr = np.frombuffer(rgb_image.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image
   
def get_filtered_pcl(pcl):
    filtered_data = []
    for i in range(4):
        filtered_data.append(pcl[i][0:5500])
    return filtered_data

def aprrox_sync_callback(lidar, rgb, odom):
    pos = odom.pose.pose.position
    cmd_vel = odom.twist.twist
    # This function is called at 10Hz
    # Subsampling at each 5th second approx
    if counter['sub-sampler'] % 6 == 0:
        # TODO: these 4 values will be pickled at index counter['index'] except image
        img = store_image(rgb)
        
        image_history.append(img)

       
        
        if len(image_history) > 4:
            image_history.pop(0)

        

        point_cloud = get_lidar_points(lidar)
        print("before append len image hisotry", len(pcl_history))

        pcl_history.append(point_cloud)

        print("before len image hisotry", len(pcl_history))
        if len(pcl_history) > 4:
            pcl_history.pop(0)

        print("after len image hisotry", len(pcl_history))
        prev_cmd_vel = get_prev_cmd_val()
        # prev_cmd_vel.pop()
        
        

        if len(image_history) == 4:

            filtered_pcl = get_filtered_pcl(pcl_history)
            print(len(prev_cmd_vel))
            align_content = {
                "pcl": filtered_pcl,
                "images": image_history,
                "prev_cmd_vel": prev_cmd_vel,
                "local_goal": None
            }

            transformer = ApplyTransformation(align_content)
            stacked_images, pcl, local_goal, prev_cmd_vel =  transformer.__getitem__(0)
            
            ckpt = torch.load("/home/ranjan/Workspace/my_works/fusion-network/scripts/fusion_model_at_val_loss_0.7496246003856262.pth")
            model = BcFusionModel()
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device)
            model.eval()
            
            pred_fusion, pred_img, pred_pcl = model(stacked_images, pcl, local_goal, prev_cmd_vel)
            print(pred_fusion)                
            msg = Twist()
            msg.linear.x = pred_fusion[0]
            msg.angular.z = pred_fusion[1]

            print('publishing')
            cmd_publisher.publish(msg)
        
        
        # previous_rbt_location.append((pos.x, pos.y, counter['index']))
        counter['index'] += 1
        counter['sub-sampler'] = 1

    counter['sub-sampler'] += 1


def lidar_testing(lidar):
    point_cloud = []
    for point in pc2.read_points(lidar, skip_nans=True, field_names=('x', 'y', 'z')):
        pt_x = point[0]
        pt_y = point[1]
        pt_z = point[2]
        # print(pt_x, pt_y, pt_z)
        point_cloud.append([pt_x, pt_y, pt_z])
    
    return point_cloud



rospy.init_node('listen_record_data', anonymous=True)


cmd_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)

lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
rgb = message_filters.Subscriber('/zed_node/rgb/image_rect_color/compressed', CompressedImage)
odom = message_filters.Subscriber('zed_node/odom', Odometry)
ts = message_filters.ApproximateTimeSynchronizer([lidar, rgb, odom], 100, 0.05, allow_headerless=True)

ts.registerCallback(aprrox_sync_callback)
odom.registerCallback(odom_callback)



rospy.spin()


