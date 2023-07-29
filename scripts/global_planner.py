#!/usr/bin/env python
# coding: utf-8
from turtle import pos
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

from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from sensor_msgs.msg import PointCloud2, CompressedImage 
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from pathlib import Path
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

from model_builder.pcl.pcl_head import PclMLP
from transformer import ApplyTransformation


device = "cuda:1" if torch.cuda.is_available() else "cpu"

print(f'Using device ========================================>  {device}')

model = PclMLP()
model.to(device)
ckpt = torch.load('/home/ranjan/Workspace/my_works/fusion-network/scripts/tf_way_pts2_model_at_130_0.016991762869687396.pth')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

content = None
path = '/home/ranjan/Workspace/my_works/fusion-network/recorded-data/train/138184_lc_sw_sc/snapshot.pickle'

with open(path, 'rb') as data:
    content = pickle.load(data)
v = 'local_goal'
print(f'content loaded: {content[0][v]}')

counter = {'index': 0, 'sub-sampler': 1}
previous_rbt_location = []
previous_velocities = []
play_back_snapshot = {}
image_history = []
pcl_history = []

def get_transformation_matrix(position, quaternion):
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    robo_coordinate_in_glob_frame  = np.array([[np.cos(theta), -np.sin(theta), position[0]],
                    [np.sin(theta), np.cos(theta), position[1]],
                    [0, 0, 1]])
    return robo_coordinate_in_glob_frame



def marker_callback(xs, ys):
    marker = Marker()
    marker.header.frame_id='base_link'
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    # marker.lifetime = rospy.Duration(1)
    marker.scale.x = 0.02
    marker.color.a = 1.0
    marker.color.r = 1
    marker.color.g = 0.0
    marker.color.b = 1
    marker.pose.orientation = Quaternion(0,0,0,1)

    points_list = []
    points_list.append(Point(y=0,x= 0,z=0))

    for i in range(5):    
        points_list.append(Point(y=ys[i],x= xs[i],z=0))

    marker.points.extend(points_list)
    pub.publish(marker)
    # key = 'index'
    # print(f'published:   {counter[key]}')
    # counter[key] += 1
    # if counter[key] > 10:
    #     counter[key] = 0
    return

def marker_gt(xs, ys):
    marker = Marker()
    marker.header.frame_id='base_link'
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    # marker.lifetime = rospy.Duration(1)
    marker.scale.x = 0.02
    marker.color.a = 1.0
    marker.color.r = 0
    marker.color.g = 1.0
    marker.color.b = 0
    marker.pose.orientation = Quaternion(0,0,0,1)

    points_list = []
    points_list.append(Point(y=0,x= 0,z=0))

    for i in range(5):    
        points_list.append(Point(y=ys[i],x= xs[i],z=0))

    marker.points.extend(points_list)
    pub_gt.publish(marker)
    # key = 'index'
    # print(f'published:   {counter[key]}')
    # counter[key] += 1
    # if counter[key] > 10:
    #     counter[key] = 0
    return

def get_goals(pts, way_pts):
    goals = pts.detach().cpu().numpy()[0]
    x = goals[:5]
    y = goals[5:]

    wx = way_pts[0,:]
    wy = way_pts[1,:]

    marker_gt(wx,wy)
    marker_callback(x,y)


def get_lidar_points(lidar):
    point_cloud = []
    for point in pc2.read_points(lidar, skip_nans=True, field_names=('x', 'y', 'z')):
        pt_x = point[0]
        pt_y = point[1]
        pt_z = point[2]

        if point[0] >= -1 and point[0] <= 5 and point[1]>=-3 and point[1]<=3 and point[2] >= 0.0299 and point[2] <= 6.0299:
            point_cloud.append([pt_x, pt_y, pt_z])

    # print(len(point_cloud))

    return point_cloud


def read_image(rgb_image):
    np_arr = np.frombuffer(rgb_image.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def aprrox_sync_callback(lidar, rgb, odom):
    pos = odom.pose.pose.position
    orientation = odom.pose.pose.orientation 
    robot_position = [pos.x, pos.y, pos.z]
    robot_orientation = [orientation.x,orientation.y, orientation.z,orientation.w]

    # print("calling  subsampler")
    if counter['sub-sampler'] % 2 == 0:
        # TODO: these 4 values will be pickled at index counter['index'] except image
        img = read_image(rgb)
        point_cloud = get_lidar_points(lidar)

        print(robot_position)
        print(content[counter['index']]['local_goal'][-1])
        print(content[counter['index']]['robot_position'][0])

        align_content = {
            "pcl": point_cloud,
            "images": [img],
            "local_goal": content[counter['index']]['local_goal'],
            "robot_pos": (robot_position, robot_orientation)
        }

        transformer = ApplyTransformation(align_content)
        # print("transformed")
        pcl, local_goal, way_pts = transformer.__getitem__(0)
        pcl = pcl.to(device)        
        local_goal = local_goal.to(device)
        pcl = pcl.unsqueeze(0)
        local_goal = local_goal.unsqueeze(0)
        with torch.no_grad():
            # print("in")
            pts = model(pcl,local_goal)
            # print("out")
            get_goals(pts/150, way_pts/150)
            counter['index'] += 1
            print(counter['index'])            
            counter['sub-sampler'] = 0

    counter['sub-sampler'] += 1



rospy.init_node('listen', anonymous=True)


cmd_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)

odom = message_filters.Subscriber('/odom', Odometry)
lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
rgb = message_filters.Subscriber('/image_raw/compressed', CompressedImage)
pub = rospy.Publisher('/world_point', Marker, queue_size=10)
pub_gt = rospy.Publisher('/world_point_gt', Marker, queue_size=10)

ts = message_filters.ApproximateTimeSynchronizer([lidar, rgb, odom], 100, 0.05, allow_headerless=True)
ts.registerCallback(aprrox_sync_callback)

# odom.registerCallback(odom_callback)



rospy.spin()


