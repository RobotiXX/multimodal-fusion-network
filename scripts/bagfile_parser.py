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

from sensor_msgs.msg import PointCloud2, CompressedImage
from nav_msgs.msg import Odometry
from pathlib import Path

counter = {'index': 0, 'sub-sampler': 1}
previous_rbt_location = []
local_goal = {}
previous_velocities = []
play_back_snapshot = {}

rosbag_path = sys.argv[1]

bag_file_name = rosbag_path.split('/')[-1]

record_storage_path = os.path.join('../recorded-data', bag_file_name)        
os.makedirs(record_storage_path, exist_ok=True)


def odom_callback(odom):
    position = odom.pose.pose.position
    cmd_vel = odom.twist.twist

    # store 20 latest liner and angular velocities
    previous_velocities.insert(0, (cmd_vel.linear.x, cmd_vel.angular.y))
    if len(previous_velocities) > 20:
        previous_velocities.pop()

    for idx, robot_pos in enumerate(previous_rbt_location):
        # if the current odom position is with in 10meter radius of a previous position
        # then use current point as a local goal of the previous position(robot's prev location)
        # TODO: should be done as a perpendicular intersection instead of delta approximation of 2

        if 0 < ((robot_pos[0] - position.x) ** 2 + (robot_pos[1] - position.y) ** 2 - 100) <= 2:
            play_back_snapshot[robot_pos[2]]["local_goal"] = (position.x, position.y)
            # print("local goal foudn for index", robot_pos[2])
            del previous_rbt_location[idx]
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
    image = cv2.resize(image, (264, 300))
    image_path = os.path.join(record_storage_path, str(idx)+".jpg")
    cv2.imwrite(image_path, image)


def aprrox_sync_callback(lidar, rgb, odom):
    pos = odom.pose.pose.position
    # This function is called at 10Hz
    # Subsampling at each 5th second approx
    if counter['sub-sampler'] % 6 == 0:
        # TODO: these 4 values will be pickled at index counter['index'] except image
        store_image(rgb, counter['index'])
        point_cloud = get_lidar_points(lidar)
        prev_cmd_vel = get_prev_cmd_val()
        robot_pos = (pos, odom.pose.pose.orientation, counter['index'])
        play_back_snapshot[counter['index']] = {
            "point_cloud": point_cloud,
            "prev_cmd_vel": prev_cmd_vel,
            "robot_position": robot_pos
        }
        previous_rbt_location.append((pos.x, pos.y, counter['index']))

        counter['index'] += 1
        counter['sub-sampler'] = 1

    counter['sub-sampler'] += 1


rospy.init_node('listen_record_data', anonymous=True)


rosbag_play_process = subprocess.Popen(
    ['rosbag', 'play', '--clock', rosbag_path, '-u', '6'])

lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
rgb = message_filters.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage)
odom = message_filters.Subscriber('/jackal_velocity_controller/odom', Odometry)
ts = message_filters.ApproximateTimeSynchronizer([lidar, rgb, odom], 100, 0.05, allow_headerless=True)

ts.registerCallback(aprrox_sync_callback)
odom.registerCallback(odom_callback)



while not rospy.is_shutdown():
    if rosbag_play_process.poll() is not None:
        print('Rosbag play has stopped, saving the data at:', record_storage_path)                
        pickle_file = os.path.join(record_storage_path, "snapshot.pickle")
        
        with open(pickle_file, 'wb') as handle:
            pickle.dump(play_back_snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
        exit(0)
