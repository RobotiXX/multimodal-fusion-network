#!/usr/bin/env python
# coding: utf-8

# In[12]:


import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CompressedImage
import cv2
import time
import rospy
import subprocess
import message_filters


# In[8]:


# bag = rosbag.Bag('../135970')


# In[9]:


point_cloud = []
def convert_pc_msg_to_np(message, t):
    for point in pc2.read_points(message, skip_nans=True):
            pt_x = point[0]
            pt_y = point[1]
            pt_z = point[2]
            point_cloud.append([pt_x,pt_y,pt_z])


# In[10]:


counter = 0
# for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
#     counter += 1
#     convert_pc_msg_to_np(msg, t)
#     print('point: ', counter)


# In[11]:


# bag.close()


# In[ ]:





# In[6]:


# rosbag_play_process = subprocess.Popen(
#         ['rosbag', 'play', '../135970', '-r', '1.0', '--clock'])


# In[13]:

rospy.init_node('listen_record_data', anonymous=True)

def callback(lidar, rgb):
    print('found lidar and rgb data')
    print(lidar.header)
    print(rgb.header)


rosbag_play_process = subprocess.Popen(
        ['rosbag', 'play', '../135970', '-r', '1.0', '--clock'], 50)


lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
rgb = message_filters.Subscriber('/left/image_color/compressed', CompressedImage)
ts = message_filters.ApproximateTimeSynchronizer([lidar, rgb], 100, 0.05, allow_headerless=True)
ts.registerCallback(callback)

rospy.spin()


# In[ ]:




