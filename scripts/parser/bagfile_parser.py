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

from sensor_msgs.msg import PointCloud2, CompressedImage, Joy
from nav_msgs.msg import Odometry
from pathlib import Path
from collections import OrderedDict

counter = {'index': 0, 'sub-sampler': 1}
previous_rbt_location = OrderedDict()
local_goal = {}
previous_velocities = []
play_back_snapshot = {}

rosbag_path = sys.argv[1]

bag_file_name = rosbag_path.split('/')[-1]

record_storage_path = os.path.join('../../recorded-data', bag_file_name)        
os.makedirs(record_storage_path, exist_ok=True)

def getJoystickValue(x, scale, kDeadZone=0.02):
    if kDeadZone != 0.0 and abs(x) < kDeadZone:
        return 0.0
    return ((x - np.sign(x) * kDeadZone) / (1.0 - kDeadZone) * scale)

def odom_callback(odom):
    position = odom.pose.pose.position
    
    keys = list(previous_rbt_location.keys())
    for key in keys:
        # if the current odom position is with in 10 meter radius of a previous position
        # then use current point as a local goal of the previous position(robot's prev location)
        # TODO: should be done as a perpendicular intersection instead of delta approximation of 2

        if ((previous_rbt_location[key][0] - position.x) ** 2 + (previous_rbt_location[key][1] - position.y) ** 2 ) >= 0.2299:
            if  len(play_back_snapshot[key]["local_goal"]) < 12:
                play_back_snapshot[key]["local_goal"].append((position.x, position.y))
                previous_rbt_location[key][0] = position.x
                previous_rbt_location[key][1] = position.y
                # break
            else:    # print("local goal foudn for index", robot_pos[2])                
                previous_rbt_location.pop(key)
        else:
            break

def get_lidar_points(lidar):
    point_cloud = []
    for point in pc2.read_points(lidar, skip_nans=True, field_names=('x', 'y', 'z')):
        pt_x = point[0]
        pt_y = point[1]
        pt_z = point[2]
        point_cloud.append([pt_x, pt_y, pt_z])
    return point_cloud


def get_prev_cmd_val():
    return previous_velocities


def store_image(rgb_image, idx):
    np_arr = np.frombuffer(rgb_image.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # image = cv2.resize(image, (264, 300))
    image_path = os.path.join(record_storage_path, str(idx)+".jpg")
    cv2.imwrite(image_path, image)

def joy_call_back(joy):
    joy_axes = joy.axes
    previous_velocities.insert(0, (
            getJoystickValue(joy_axes[4], -1.6),
            getJoystickValue(joy_axes[3], -1.6),
            getJoystickValue(joy_axes[0], -np.deg2rad(90.0), kDeadZone=0.0),
        ))
    while len(previous_velocities) > 22:
        previous_velocities.pop()

def aprrox_sync_callback(lidar, rgb, odom, joy):
    pos = odom.pose.pose.position
    cmd_vel = odom.twist.twist
    joy_axes = joy.axes
    # This function is called at 10Hz
    # Subsampling at each 5th second approx
    # print(f'counter: {counter["sub-sampler"]}')
    if counter['sub-sampler'] % 2 == 0:
        # TODO: these 4 values will be pickled at index counter['index'] except image
        store_image(rgb, counter['index'])
        point_cloud = get_lidar_points(lidar)
        prev_cmd_vel = get_prev_cmd_val().copy()
        prev_cmd_vel.pop(0)
        prev_cmd_vel.pop(0)
        orientation = odom.pose.pose.orientation
        robot_pos = ([pos.x, pos.y, pos.z], [orientation.x, orientation.y, orientation.z, orientation.w], counter['index'])
        # Record data at current point
        gt_cmd_vel = (
            getJoystickValue(joy_axes[4], -1.6),
            getJoystickValue(joy_axes[3], -1.6),
            getJoystickValue(joy_axes[0], -np.deg2rad(90.0), kDeadZone=0.0),
        )
        # print(f'ground truth velocity: {gt_cmd_vel}\n\n')
        # print("here")
        play_back_snapshot[counter['index']] = {
            "point_cloud": point_cloud,
            # "prev_cmd_vel": prev_cmd_vel,
            "robot_position": robot_pos,
            "gt_cmd_vel": gt_cmd_vel,
            "local_goal": []
        }
        previous_rbt_location[counter['index']] =[pos.x, pos.y]

        counter['index'] += 1
        counter['sub-sampler'] = 0

    counter['sub-sampler'] += 1    


rospy.init_node('listen_record_data', anonymous=True)


rosbag_play_process = subprocess.Popen(
    ['rosbag', 'play', '--clock', rosbag_path])

lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
rgb = message_filters.Subscriber('/image_raw/compressed', CompressedImage)
odom = message_filters.Subscriber('/odom', Odometry)
joy = message_filters.Subscriber('/joystick', Joy)



odom.registerCallback(odom_callback)
joy.registerCallback(joy_call_back)

ts = message_filters.ApproximateTimeSynchronizer([lidar, rgb, odom, joy], 100, 0.05, allow_headerless=True)
ts.registerCallback(aprrox_sync_callback)

while not rospy.is_shutdown():
    if rosbag_play_process.poll() is not None:
        print('Rosbag play has stopped, saving the data at:', record_storage_path)                
        pickle_file = os.path.join(record_storage_path, "snapshot.pickle")
        
        with open(pickle_file, 'wb') as handle:
            pickle.dump(play_back_snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
        exit(0)
