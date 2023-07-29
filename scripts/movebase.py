#!/usr/bin/env python

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Path, Odometry
import rospy
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

class Robot_config():
    def __init__(self):
        self.X = 0 # inertia frame
        self.Y = 0
        self.PSI = 0
        self.global_path = []
        self.gx = 0 # body frame
        self.gy = 0
        self.gp = 0
        self.los = 3
        # self.los = 5
    
    def get_robot_status(self, msg):
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        q0 = msg.pose.pose.orientation.w
        self.X = msg.pose.pose.position.x
        self.Y = msg.pose.pose.position.y
        self.PSI = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
        #print(self.X, self.Y, self.PSI)
    
    def get_global_path(self, msg):
        #print(msg.poses)
        # self.global_path = []
        gp = []
        for pose in msg.poses:
            gp.append([pose.pose.position.x, pose.pose.position.y])
            # self.global_path.append([pose.pose.position.x, pose.pose.position.y])
        #print(len(self.global_path))
        gp = np.array(gp)
        x = gp[:,0]
        # xhat = x
        try:
            xhat = scipy.signal.savgol_filter(x, 19, 3)
        except:
            xhat = x
        y = gp[:,1]
        # yhat = y
        try: 
            yhat = scipy.signal.savgol_filter(y, 19, 3)
        except: 
            yhat = y
        # plt.figure()
        # plt.plot(xhat, yhat, 'k', linewidth=1)
        # plt.axis('equal')
        # plt.savefig("/home/xuesu/gp_plt.png")
        # plt.close()
        gphat = np.column_stack((xhat, yhat))
        gphat.tolist()
        self.global_path = gphat
        # print(self.global_path)



def transform_lg(wp, X, Y, PSI):
    R_r2i = np.matrix([[np.cos(PSI), -np.sin(PSI), X], [np.sin(PSI), np.cos(PSI), Y], [0, 0, 1]])
    R_i2r = np.linalg.inv(R_r2i)
    #print(R_r2i)
    pi = np.matrix([[wp[0]], [wp[1]], [1]])
    pr = np.matmul(R_i2r, pi)
    #print(pr) 
    lg = np.array([pr[0,0], pr[1, 0]])
    #print(lg)
    return lg
    
        
if __name__ == '__main__':
    robot_config = Robot_config()
    rospy.init_node('jackal_global_plan', anonymous=True)
    sub_robot = rospy.Subscriber("/odometry/filtered", Odometry, robot_config.get_robot_status)
    sub_gp = rospy.Subscriber("/move_base/TrajectoryPlannerROS/global_plan", Path, robot_config.get_global_path)

    lg = Pose()
    pub_lg = rospy.Publisher('local_goal', Pose, queue_size=10)
    while not rospy.is_shutdown():
        gp = robot_config.global_path
        X = robot_config.X
        Y = robot_config.Y
        PSI = robot_config.PSI
        los = robot_config.los
        
        #if len(gp)==0:
        lg_x = 0
        lg_y = 0
        #else:
        if len(gp)>0:
            lg_flag = 0
            for wp in gp:
                dist = (np.array(wp)-np.array([X, Y]))**2
                dist = np.sum(dist, axis=0)
                dist = np.sqrt(dist)
                if dist > los:
                    lg_flag = 1
                    lg = transform_lg(wp, X, Y, PSI)
                    lg_x = lg[0]
                    lg_y = lg[1]
                    break
            if lg_flag == 0:
                lg = transform_lg(gp[-1], X, Y, PSI)
                lg_x = lg[0]
                lg_y = lg[1]
                
        # print(lg_x, lg_y)
        local_goal = Pose()
        local_goal.position.x = lg_x
        local_goal.position.y = lg_y
        local_goal.orientation.w = 1
        pub_lg.publish(local_goal)
        
