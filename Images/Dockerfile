FROM ubuntu:20.04

RUN apt-get update

ENV TZ=US/Eastern

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update

RUN apt-get install -y python3

RUN apt-get update

RUN apt-get install  \
    -y  \
    lsb-release

RUN apt-get clean all

RUN apt install  \
    -y \
    curl

RUN apt install  \
    wget


RUN apt-get update

RUN apt-get install \
    -y  \
    gnupg2

RUN sh  \
    -c  \
    'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'



#RUN curl -s \
#    https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN wget  \
    http://packages.ros.org/ros.key


RUN apt-key add  \
    ros.key

RUN apt update

RUN apt install  \
    -y \
    ros-noetic-desktop-full

RUN apt update

RUN apt install  \
    -y \
    python3-rosdep  \
    python3-rosinstall  \
    python3-rosinstall-generator  \
    python3-wstool build-essential

RUN apt install  \
    -y \
    python3-rosdep

RUN rosdep init

RUN rosdep update

RUN echo  \
    ". /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN git clone https://github.com/ros/ros_comm


RUN apt install \
    -y \
     python3-pip

WORKDIR ./ros_comm/tools/rosbag
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/clients/rospy
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/utilities/roslz4
RUN pip install .
WORKDIR /


WORKDIR ./ros_comm/tools/rosnode
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/tools/rosmsg
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/tools/rosnode
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/tools/rosparam
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/tools/rosservice
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/tools/roslaunch/
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/tools/rosmaster/
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/tools/rosnode
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/tools/topic_tools
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/tools/rostopic
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/tools/rostest
RUN pip install .
WORKDIR /

WORKDIR ./ros_comm/utilities/roswtf
RUN pip install .
WORKDIR /


WORKDIR ./ros_comm/utilities/message_filters
RUN pip install .
WORKDIR /

RUN yes|  \
    pip install \
    pandas

