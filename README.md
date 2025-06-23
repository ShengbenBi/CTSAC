# CTSAC: Curriculum-Based Transformer Soft Actor-Critic for Goal-Oriented Robot Exploration
With the increasing demand for efficient and flexible robotic exploration solutions, Reinforcement Learning (RL) is becoming a promising approach in the field of autonomous robotic exploration. However, current RL-based exploration algorithms often face limited environmental reasoning capabilities, slow convergence rates, and substantial challenges in Sim-To-Real (S2R) transfer. To address these issues, we propose a Curriculum Learning-based Transformer Reinforcement Learning Algorithm (CTSAC) aimed at improving both exploration efficiency and transfer performance. To enhance the robot's reasoning ability, a Transformer is integrated into the perception network of the Soft Actor-Critic (SAC) framework, leveraging historical information to improve the farsightedness of the strategy. A periodic review-based curriculum learning is proposed, which enhances training efficiency while mitigating catastrophic forgetting during curriculum transitions.
Training is conducted on the ROS-Gazebo continuous robotic simulation platform, with LiDAR clustering optimization to further reduce the S2R gap. Experimental results demonstrate the CTSAC algorithm outperforms the state-of-the-art non-learning and learning-based algorithms in terms of success rate and success rate-weighted exploration time. Moreover, real-world experiments validate the strong S2R transfer capabilities of CTSAC.
This project has been accepted by 2025 ICRA. Address:https://arxiv.org/abs/2503.14254

https://github.com/202409230100_x264.mp4

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

