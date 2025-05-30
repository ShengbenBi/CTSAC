import math                 
import os                   
import random               
import subprocess           
import time                 
import json                 
from os import path         

import numpy as np                              
import rospy                                    
import sensor_msgs.point_cloud2 as pc2          
from gazebo_msgs.msg import ModelState          
from gazebo_msgs.srv import DeleteModel, GetWorldProperties
from geometry_msgs.msg import Twist             
from nav_msgs.msg import Odometry               
from sensor_msgs.msg import PointCloud2         
from squaternion import Quaternion              
from std_srvs.srv import Empty                  
from visualization_msgs.msg import Marker       
from visualization_msgs.msg import MarkerArray  
from parameter import get_parameters
import roslaunch
import argparse
from multiprocessing import Process
from geometry_msgs.msg import Pose
from PIL import Image
import yaml
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import DeleteModelRequest

GOAL_REACHED_DIST = 0.45   
COLLISION_DIST    = 0.25   
TIME_DELTA        = 0.1    
x                 = 0      
y                 = 0      
previous_action   = None   


parser = get_parameters()
args = parser.parse_args()

FILTER_GROUND_HEIGHT = -0.072

class GazeboEnv:
    def __init__(self, launchfile, environment_dim, robot_num):
        self.environment_dim      = environment_dim                        
        self.robot_num            = robot_num                              
        self.odom_x               = 0                                      
        self.odom_y               = 0                                      
        self.goal_x               = 1.0                                    
        self.goal_y               = 0.0                                    
        self.upper                = 9.5                                    
        self.lower                = -9.5                                   
        self.velodyne_data        = np.ones(self.environment_dim) * 10     
        self.goal_reach_dist      = GOAL_REACHED_DIST                      
        self.current_map_level    = 0                                      
        self.map_index_update     = False                                  
        self.map_indices          = []                                     
        self.probabilities        = []                                     
        self.models               = ["map0"]                               
        self.old_models           = []                                     
        self.distance_threshold   = args.distance_threshold                
        self.collect_position     = []                                     
        self.repeat_position      = 0                                      
        self.create_cardboard_box = True                                   
        self.last_map_index       = 1                                      
        self.map_index            = 0                                      
        self.test_mode            = 0                                      
        self.set_self_state = [ModelState() for _ in range(self.robot_num)]  

        for i in range(self.robot_num):
            self.set_self_state[i].model_name         = 'r' + str(i)
            self.set_self_state[i].pose.position.x    = 0.0
            self.set_self_state[i].pose.position.y    = 0.0
            self.set_self_state[i].pose.position.z    = 0.0
            self.set_self_state[i].pose.orientation.x = 0.0  
            self.set_self_state[i].pose.orientation.y = 0.0
            self.set_self_state[i].pose.orientation.z = 0.0
            self.set_self_state[i].pose.orientation.w = 1.0

            setattr(self, f"last_odom_{i}", None)
            setattr(self, f"velodyne_data_{i}", None)
        
        self.gaps = [[-np.pi / 2 - np.pi * 2  / (self.environment_dim * 3) , -np.pi / 2 + np.pi * 2  / (self.environment_dim * 3)]]
        for m in range(self.environment_dim - 1):
            if m < self.environment_dim * 3 / 4:
                self.gaps.append([self.gaps[m][1], self.gaps[m][1] + np.pi * 4  / (self.environment_dim * 3)])
            else:
                self.gaps.append([self.gaps[m][1], self.gaps[m][1] + (12*self.environment_dim*np.pi - 16*np.pi) /(3*self.environment_dim **2 - 12* self.environment_dim)])

        port = "11311"                              
        subprocess.Popen(["roscore", "-p", port])   
        print("Roscore launched!")   

        #load map
        json_file_path = "../catkin_ws/src/turtlebot3/turtlebot3_bringup/cl_models/cl_map.json" 
        if not os.path.exists(json_file_path):       
            raise IOError("File " + json_file_path + " does not exist")
        else:
            self.load_map(json_file_path)
            print("json Map loaded!")

        #launch gazebo
        rospy.init_node("gym", anonymous=True) 
        if launchfile.startswith("/"):          
            fullpath = launchfile               
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)  
        if not path.exists(fullpath):                                                 
            raise IOError("File " + fullpath + " does not exist")                     

        subprocess.Popen(["roslaunch", "-p", port, fullpath])      
        
        for i in range(self.robot_num):  
            robot_name = 'r' + str(i)
            model = 'waffle'
            position = '-x 0.0 -y 0.0 -z 0.01 -R 0 -P 0 -Y +0.0'
            collide_bitmask = '0x{:02x}'.format(1 << i)  
            self.create_robot_launch(robot_name, model, position, collide_bitmask)
            time.sleep(3)
            
        print("Gazebo launched!")                                   


        for i in range(self.robot_num): 
            setattr(self, f"vel_pub_{i}",  rospy.Publisher (f"/r{i}/cmd_vel", Twist, queue_size=1))
            setattr(self, f"odom_{i}",     rospy.Subscriber(f"/r{i}/odom", Odometry,               self.create_odom_callback(i),     queue_size=1))            
            setattr(self, f"velodyne_{i}", rospy.Subscriber(f"/r{i}/velodyne_points", PointCloud2, self.create_velodyne_callback(i), queue_size=1))
        self.set_state   = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)                   
        self.publisher   = rospy.Publisher("goal_point", MarkerArray, queue_size=3)                               
        self.publisher2  = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)            
        self.publisher3  = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)         
        self.unpause     = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)                                   
        self.pause       = rospy.ServiceProxy("/gazebo/pause_physics", Empty)                                     
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)                                       

        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    def load_map(self, filename):
        with open(filename, 'r') as f:
            self.objects = json.load(f)

    def create_robot_launch(self, robot_name, model, position, collide_bitmask):
        def launch_robot():
            launch_file_path = "/home/jetson/SAC-robot-navigation/catkin_ws/src/turtlebot3/turtlebot3_bringup/launch/turtlebot3_model.launch"
            port             = "11311" 
            cli_args         = [launch_file_path, 'model:={}'.format(model), 'multi_robot_name:={}'.format(robot_name), 'robot_position:={}'.format(position), 'collide_bitmask:={}'.format(collide_bitmask)]
            command          = ["roslaunch", "-p", port] + cli_args
            process          = subprocess.Popen(command)
        p = Process(target=launch_robot)
        p.start()

    def create_velodyne_callback(self, robot_id):                                          
        def velodyne_callback(v):                                                          
            data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z"))) 
            velodyne_data = np.ones(self.environment_dim) * 10                             
            for i in range(len(data)):                                                     
                if data[i][2] > FILTER_GROUND_HEIGHT:                                        
                    dot = data[i][0] * 1 + data[i][1] * 0                                  
                    mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))    
                    mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))                      
                    beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])            
                    dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)  
                    for j in range(len(self.gaps)):
                        if self.gaps[j][0] <= beta < self.gaps[j][1]:                      
                            velodyne_data[j] = min(velodyne_data[j], dist)                 
                            break
            setattr(self, f"velodyne_data_{robot_id}", velodyne_data)
        return velodyne_callback
    
    def create_odom_callback(self, robot_id):                 
        def odom_callback(od_data):
            setattr(self, f"last_odom_{robot_id}", od_data)
        return odom_callback

    def create_map(self, test_mode = 0):  
        if test_mode == 0:
            if self.create_cardboard_box:
                self.create_cardboard_box = False
                for obj in self.objects["cardboard_box"]:             
                    with open(obj['file'], 'r') as f:
                        model_xml = f.read()
                    initial_pose               = Pose()
                    initial_pose.position.x    = obj['x']
                    initial_pose.position.y    = obj['y']
                    initial_pose.position.z    = obj['z']
                    initial_pose.orientation.x = obj['qx']
                    initial_pose.orientation.y = obj['qy']
                    initial_pose.orientation.z = obj['qz']
                    initial_pose.orientation.w = obj['qw']
                    self.spawn_model(obj['name'], model_xml, '', initial_pose, "world")

            if self.map_index_update and self.current_map_level < 6:  
                self.current_map_level += 1
                self.map_indices        = list(range(self.current_map_level + 1))
                self.probabilities      = [(i + 1)**2 for i in self.map_indices]
                total                   = sum(self.probabilities)
                self.probabilities      = [p / total for p in self.probabilities]  
                self.map_index_update   = False
            map_index = 0 if self.current_map_level == 0 else np.random.choice(self.map_indices, p=self.probabilities)

        else:
            map_index = test_mode

        if self.last_map_index != map_index:
            self.last_map_index = map_index
            self.old_models = self.models.copy()            
            self.models = []                                

            map_key = "map" + str(map_index)                
            for obj in self.objects[map_key]:               
                with open(obj['file'], 'r') as f:
                    model_xml = f.read()

                initial_pose               = Pose()
                initial_pose.position.x    = obj['x']
                initial_pose.position.y    = obj['y']
                initial_pose.position.z    = obj['z']
                initial_pose.orientation.x = obj['qx']
                initial_pose.orientation.y = obj['qy']
                initial_pose.orientation.z = obj['qz']
                initial_pose.orientation.w = obj['qw']

                self.spawn_model(obj['name'], model_xml, '', initial_pose, "world")
                self.models.append(obj['name'])  

        return map_index

    def step(self, action, robot_id=0):
        done = False
        # Publish the robot action
        vel_cmd = Twist()              
        vel_cmd.linear.x = action[0]    
        vel_cmd.angular.z = action[1]

        getattr(self, f"vel_pub_{robot_id}").publish(vel_cmd)      
        self.publish_markers(action)                               

        rospy.wait_for_service("/gazebo/unpause_physics")          
        try:                                                       
            self.unpause()                                         
        except (rospy.ServiceException) as e:                      
            print("/gazebo/unpause_physics service call failed")  

        time.sleep(TIME_DELTA)                                     

        rospy.wait_for_service("/gazebo/pause_physics")            
        try:
            pass                                                    
            self.pause()                                            
        except (rospy.ServiceException) as e:                       
            print("/gazebo/pause_physics service call failed")     

        
        collision, min_laser = self.observe_collision(getattr(self, f"velodyne_data_{robot_id}"))
        v_state = []                                                      
        v_state[:] = getattr(self, f"velodyne_data_{robot_id}")[:]        
        laser_state = [v_state]                                           

        self.odom_x = getattr(self, f"last_odom_{robot_id}").pose.pose.position.x                 
        self.odom_y = getattr(self, f"last_odom_{robot_id}").pose.pose.position.y                 

        
        for pos in self.collect_position:
        
            distance = abs(self.odom_x - pos[0]) + abs(self.odom_y - pos[1])
            
            if distance < self.distance_threshold:
                self.repeat_position += 1
        self.collect_position.append([self.odom_x, self.odom_y])

        quaternion = Quaternion(                                                                  
            getattr(self, f"last_odom_{robot_id}").pose.pose.orientation.w, 
            getattr(self, f"last_odom_{robot_id}").pose.pose.orientation.x,
            getattr(self, f"last_odom_{robot_id}").pose.pose.orientation.y,
            getattr(self, f"last_odom_{robot_id}").pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)                                                
        angle = round(euler[2], 4)                                                                

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        beta = np.arctan2(self.goal_y - self.odom_y, self.goal_x - self.odom_x)
        theta = beta - angle
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        '''
        if self.goal_reach_dist >=0.2:
            self.goal_reach_dist = self.goal_reach_dist - 0.000002
            # self.goal_reach_dist = 0.35
        '''
        if distance < self.goal_reach_dist:
            done = True             

        robot_state = [distance, theta, action[0], action[1]]                                                #机器人状态
        state = np.append(laser_state, robot_state)                                                          #将laser_state和robot_state拼接在一起，append()函数用于在列表末尾添加新的对象
        reward = self.get_reward(done, collision, action, min_laser, distance, self.repeat_position)              #计算奖励
        
        self.repeat_position = 0          

        return state, reward, done, collision                                                                #返回状态、奖励、是否结束、是否到达目标

    def reset(self, robot_x=None, robot_y=None, target_x=None, target_y=None):

        
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()                          
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        
        self.map_index = self.create_map(self.test_mode)     

        if self.old_models:
            for model in self.old_models:
                try:
                    delete_model_request = DeleteModelRequest(model_name=model)
                    rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)(delete_model_request)
                except rospy.ServiceException as e:
                    pass
            self.old_models = []
        
        if self.test_mode == 0:
            self.change_robot_position()
            self.change_goal()
            self.random_box()
        else : 
            
            quaternion = Quaternion.from_euler(0.0, 0.0, 0.0)
            for i in range(self.robot_num):
                
                object_state = self.set_self_state[i]
                object_state.pose.position.x = robot_x      
                object_state.pose.position.y = robot_y      
                # object_state.pose.position.z = 0.
                object_state.pose.orientation.x = quaternion.x
                object_state.pose.orientation.y = quaternion.y
                object_state.pose.orientation.z = quaternion.z
                object_state.pose.orientation.w = quaternion.w
                self.set_state.publish(object_state)
                self.odom_x = x            
                self.odom_y = y
            
            self.goal_x = target_x
            self.goal_y = target_y
        

        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

    def get_state(self, robot_id):

        self.odom_x = getattr(self, f"last_odom_{robot_id}").pose.pose.position.x                 
        self.odom_y = getattr(self, f"last_odom_{robot_id}").pose.pose.position.y                 
        quaternion = Quaternion(                                                                  
            getattr(self, f"last_odom_{robot_id}").pose.pose.orientation.w, 
            getattr(self, f"last_odom_{robot_id}").pose.pose.orientation.x,
            getattr(self, f"last_odom_{robot_id}").pose.pose.orientation.y,
            getattr(self, f"last_odom_{robot_id}").pose.pose.orientation.z,
                                )
        euler = quaternion.to_euler(degrees=False)                                                
        angle = round(euler[2], 4)                                                                

        v_state = []                                                      
        v_state[:] = getattr(self, f"velodyne_data_{robot_id}")[:]        
        laser_state = [v_state]

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        beta = np.arctan2(self.goal_y - self.odom_y, self.goal_x - self.odom_x)
        theta = beta - angle
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_robot_position(self):
        for i in range(self.robot_num):
            angle = np.random.uniform(-np.pi, np.pi)            
            quaternion = Quaternion.from_euler(0.0, 0.0, angle) 

            position_ok = False
            if args.random_robot_position:
                while not position_ok:               
                    x = np.random.uniform(-9.5 , 9.5 ) 
                    y = np.random.uniform(-9.5 , 9.5 ) 
                    position_ok = check_pose(x, y, self.map_index)    
            else:
                x = args.robot_position_x
                y = args.robot_position_y
                position_ok = check_pose(x, y, self.map_index)                         
                while not position_ok:                                
                    x = np.random.uniform(-9.5 , 9.5 )               
                    y = np.random.uniform(-9.5 , 9.5 )                   
                    position_ok = check_pose(x, y, self.map_index)               

            object_state = self.set_self_state[i]
            object_state.pose.position.x = x
            object_state.pose.position.y = y
            # object_state.pose.position.z = 0.
            object_state.pose.orientation.x = quaternion.x
            object_state.pose.orientation.y = quaternion.y
            object_state.pose.orientation.z = quaternion.z
            object_state.pose.orientation.w = quaternion.w
            self.set_state.publish(object_state)
            self.odom_x = x            
            self.odom_y = y  

    def change_goal(self):

        goal_ok = False
        while not goal_ok:
            self.goal_x = random.uniform(self.upper, self.lower)
            self.goal_y = random.uniform(self.upper, self.lower)
            goal_ok     = check_pose(self.goal_x, self.goal_y, self.map_index)

    def random_box(self):
        
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-9,  9 )
                y = np.random.uniform(-9 , 9 )
                box_ok = check_pose(x, y, self.map_index, 30)        
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])  
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])   
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):  
        markerArray = MarkerArray()
        marker = Marker()         
        
        marker.header.frame_id = "odom"        
        marker.type = marker.CYLINDER          
        marker.action = marker.ADD             
        marker.scale.x = 0.1                   
        marker.scale.y = 0.1                   
        marker.scale.z = 0.01                  
        marker.color.a = 1.0                   
        marker.color.r = 0.0                   
        marker.color.g = 1.0                   
        marker.color.b = 0.0                   
        marker.pose.orientation.w = 1.0        
        marker.pose.position.x = self.goal_x   
        marker.pose.position.y = self.goal_y   
        marker.pose.position.z = 0             

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2               = MarkerArray()
        marker2                    = Marker()
        marker2.header.frame_id    = "odom"
        marker2.type               = marker.CUBE            
        marker2.action             = marker.ADD
        marker2.scale.x            = abs(action[0])      
        marker2.scale.y            = 0.1
        marker2.scale.z            = 0.01
        marker2.color.a            = 1.0
        marker2.color.r            = 1.0
        marker2.color.g            = 0.0
        marker2.color.b            = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x    = 5
        marker2.pose.position.y    = 0
        marker2.pose.position.z    = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod           
    def observe_collision(laser_data):

        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, min_laser
        return False, min_laser

    @staticmethod           
    def get_reward(target, collision, action, min_laser, distance, repeat_position):
        global previous_action
        r3 = lambda x: 1 - x   if x < 0.4 else 0.0                      
        r4 = lambda x: x / 10  if x < 10   else 1                            
        r5 = lambda x, y:  2   if (x * y < 0 and abs(x - y) > 0.2) else 0     
        r6 = lambda x: 1       if abs(x)>0.5 else 0                           
        if target: 
            previous_action = None
            return 120.0
        elif collision:
            previous_action = None
            return -100.0
        else:
            if previous_action is None:
                reward = (-r6(abs(action[1]))/8 - r4(distance)/5  - repeat_position/5)
            else:
                reward = (-r6(abs(action[1]))/8 - r4(distance)/5  - repeat_position/5)
            previous_action = action[:]
            return reward - 0.2

def check_pose(x, y, map_index, radius=45):
    
    map_yaml_file = f'/home/jetson/SAC-robot-navigation-CL/catkin_ws/src/turtlebot3/turtlebot3_bringup/cl_models/maps/map{map_index}/map.yaml'
    # 读取yaml文件
    with open(map_yaml_file, 'r') as stream:
        map_data = yaml.safe_load(stream)

    map_file        = map_data['image']
    resolution      = map_data['resolution']                
    occupied_thresh = (1-map_data['occupied_thresh']) * 255

    map_image = Image.open(map_file)
    if map_image is None:
        raise ValueError('Invalid map level')
    
    if not (-10 <= x < 10 and -10 <= y < 10):
        return False
    
    
    x = x / resolution
    y = y / resolution

    y = map_image.height/2 - y    
    x = map_image.width /2 + x   

    left            = max(0, x - radius)
    right           = min(map_image.width, x + radius)
    top             = max(0, y - radius)
    bottom          = min(map_image.height, y + radius)
    total_pixels    = 0
    obstacle_pixels = 0
    for i in range(int(left), int(right)):
        for j in range(int(top), int(bottom)):
            pixel_value = map_image.getpixel((i, j))
            if isinstance(pixel_value, tuple):
                r, g, b, *_ = pixel_value
                pixel_value = 0.2989 * r + 0.5870 * g + 0.1140 * b
            total_pixels += 1
            if pixel_value < occupied_thresh:
                obstacle_pixels += 1
    return obstacle_pixels / total_pixels < 0.05
