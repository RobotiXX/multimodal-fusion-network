#!/usr/bin/env python
# coding: utf-8

# In[12]:



import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CompressedImage, Imu
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

rospy.init_node('listen_record_data', anonymous=True)

def callback(lidar, rgb, imu):
    print('*******Found lidar, rgb and Imu data**********')
    # print(lidar.header)
    # print(rgb.header)
    print(imu)


rosbag_play_process = subprocess.Popen(
        ['rosbag', 'play', '../133231', '-r', '1.0', '--clock'])


lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
rgb = message_filters.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage)
imu = message_filters.Subscriber('/imu/data_raw', Imu)
ts = message_filters.ApproximateTimeSynchronizer([lidar, rgb, imu], 100, 0.05, allow_headerless=True)
ts.registerCallback(callback)

rospy.spin()

# In[ ]:




