#!/usr/bin/env python
# coding: utf-8
import numpy as np
# In[12]:


import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CompressedImage, Imu
from nav_msgs.msg import Odometry
import rospy
import subprocess
import message_filters
import sensor_msgs.point_cloud2 as pc2
import cv2

# In[8]:


# bag = rosbag.Bag('../135970')


# In[9]:


point_cloud = []


def convert_pc_msg_to_np(message, t):
    for point in pc2.read_points(message, skip_nans=True):
        pt_x = point[0]
        pt_y = point[1]
        pt_z = point[2]
        point_cloud.append([pt_x, pt_y, pt_z])


# In[10]:


counter = 0

rospy.init_node('listen_record_data', anonymous=True)

d = {0: 0, 1: 1}
previous_points = []
local_goal = {}


def odom_callback(odom):
    position = odom.pose.pose.position

    # store list of liner and angular velocities
    # call some function


    # store position and detach head
    for idx, robot_pos in enumerate(previous_points):
        # print("robot pos:", robot_pos)
        if 0 < ((robot_pos[0] - position.x) ** 2 + (robot_pos[1] - position.y) ** 2 - 100) <= 2:
            local_goal[robot_pos[2]] = (position.x, position.y)
            print("current idx:", d[0])
            print("robot position:", robot_pos)
            del previous_points[idx]
            print(local_goal)
            print("\n")
            break


def get_lidar_points(lidar):
    point_cloud = []
    for point in pc2.read_points(lidar, skip_nans=True, field_names=('x', 'y', 'z')):
        pt_x = point[0]
        pt_y = point[1]
        pt_z = point[2]
        point_cloud.append([pt_x, pt_y, pt_z])

    print(len(point_cloud))
    return point_cloud

def get_image(rgb_image):
    np_arr = np.fromstring(rgb_image.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    x = str(d[0]) +".jpg"
    cv2.imwrite(x, image_np)


def callback(lidar, rgb, odom):
    pos = odom.pose.pose.position

    if d[1] % 6 == 0:
        point_cloud = get_lidar_points(lidar)
        image = get_image(rgb)
        print(len(point_cloud))
        previous_points.append((pos.x, pos.y, d[0]))
        d[0] += 1
        d[1] = 1
    d[1] += 1


rosbag_play_process = subprocess.Popen(
    ['rosbag', 'play', '--clock', '../bagfiles/133231.bag'])

lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
rgb = message_filters.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage)
odom = message_filters.Subscriber('/jackal_velocity_controller/odom', Odometry)
ts = message_filters.ApproximateTimeSynchronizer([lidar, rgb, odom], 100, 0.05, allow_headerless=True)
ts.registerCallback(callback)
# odom.registerCallback(odom_callback)
rospy.spin()

# In[ ]:
