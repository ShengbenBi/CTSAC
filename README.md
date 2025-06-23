# CTSAC

https://github.com/user-attachments/assets/bd868a88-eb39-4b8c-8aa7-665e72c878dd

This project has been accepted by 2025 ICRA. Address:https://arxiv.org/abs/2503.14254
# CTSAC-Curriculum-Based Transformer Soft Actor-Critic for Goal-Oriented Robot Exploration

## Installation

### Main dependencies：

ROS Noetic on Ubuntu 20.04

Pytorch 3.8

### 1 Clone the repository：

```
cd ~

git clone https://github.com/ShengbenBi/CTSAC/SAC-robot-navigation-CL
```

### 2 Compile the workspace:

```
cd ~/SAC-robot-navigation-CL/catkin_ws

catkin_make
```

If it prompts that a component is missing, install what is missing.

### 3 Replace Path:

Replace all **/jetson** in the file with your computer name，such as **/bsb**

### 4 Train：

```
cd ~/SAC-robot-navigation-CL/catkin_ws

source ./devel/setup.bash

cd ../SAC

python3 velodyne_train.py
```

The first run will be a bit slow because it needs to download the relevant models in gazebo, which takes about 10 minutes.

### 5 Tensorboard：

```
cd ~/SAC-robot-navigation-CL/SAC
tensorboard --logdir runs

```

### 6 Test:

```
cd ~/SAC-robot-navigation-CL/catkin_ws

source ./devel/setup.bash

cd ../SAC

python3 velodyne_test.py
```

### Kill processes：

```
killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
```
If this code is helpful to your study and research, please don't hesitate to give me your star, which will be the greatest encouragement to me.
This code is modified based on [DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation). Thanks for his open source and work.

