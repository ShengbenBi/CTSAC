import time
import numpy as np
from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv
from SAC import SAC
from parameter import get_parameters

class Velodyne_Trainer():
    def __init__(self, robot_num=1):

        parser = get_parameters()
        self.args = parser.parse_args()

        
        self.environment_dim = 72
        self.robot_dim = 4
        self.robot_num = robot_num

        self.env = GazeboEnv("multi_robot_scenario.launch", self.environment_dim, self.robot_num)

        time.sleep(3)                                                
        self.state_dim     = self.environment_dim + self.robot_dim   
        self.action_dim    = 2                                       
        self.max_action    = 1                                       
        self.max_iteration = [300, 350, 400, 500, 600, 700, 800]     
        self.file_name     = "SAC_velodyne"                          
        self.env.test_mode = 0                                       
        
        self.network = SAC(self.state_dim, self.action_dim, self.max_action)
        
        self.replay_buffer = ReplayBuffer(self.args.capacity, self.args.seed)
        
        if self.args.load: self.network.load()

    def train(self, robot_id=0):
        done                 = True
        episode_reward       = 0                        
        expl_noise           = self.args.expl_noise     
        timesteps_since_eval = 0                    
        step_count           = 0                        
        recent_successes     = []                       
        success_rate         = 0                        

        for episode in range(self.args.iteration):      
            self.env.reset()    
            state      = self.env.get_state(robot_id)   
            done       = False
            collision  = False
            step_count = 0
            state_collect = []                          

            for timesteps in range(self.max_iteration[self.env.map_index]):
                
                step_count +=1
                if expl_noise > self.args.expl_min:  
                    expl_noise = expl_noise - ((1 - self.args.expl_min) / self.args.expl_decay_steps)

                state_collect.append(state)
                '''
                #如果收集的数据多于keys_num个，删除最早的数据
                if len(state_collect) > self.args.keys_num:    #储存最近keys_num个状态
                    state_collect.pop(0)
                '''
                current_length = len(state_collect)
                state_collect_np = np.array(state_collect).reshape(1, current_length, self.state_dim)

                action = self.network.select_action(state_collect_np)  
                action = action[-1]
                noisy = np.random.normal(0, expl_noise, size=self.action_dim) / 2
                action = (action + noisy).clip(-self.max_action, self.max_action)
                a_in = [(action[0]+1)/2, action[1]]

                next_state, reward, done, collision = self.env.step(a_in)   

                done_bool = 0 if timesteps + 1 == self.max_iteration[self.env.map_index] else int(done)                   
                end       = 1 if (timesteps + 1 == self.max_iteration[self.env.map_index] or collision == 1) else int(done) 

                episode_reward += reward
                average_reward  = episode_reward / step_count
                
                self.replay_buffer.add(state, action, reward, next_state, done_bool, end)

                if self.replay_buffer.size() > self.args.batch_size*self.args.keys_num:
                    if timesteps_since_eval % self.args.update_interval == 0:  
                        timesteps_since_eval %= self.args.update_interval
                        self.network.update(self.replay_buffer, self.args.batch_size, self.env.current_map_level)   

                state = next_state  

                if end:    
                    self.env.collect_position = []
                    state_collect.pop(0)
                    break

 
            if (episode % self.args.log_interval == 0) and self.args.save:
                self.network.save()

            
            if done_bool ==1:
                recent_successes.append(1)
            else:   
                recent_successes.append(0)

            if len(recent_successes) > 100:
                recent_successes.pop(0) 
            if len(recent_successes) > 30:
                success_rate = sum(recent_successes) / len(recent_successes)
                if success_rate > 0.8:
                    self.env.map_index_update = True
                    recent_successes = []
                    self.replay_buffer.clear()
                    
                    if success_rate > 0.9 and self.env.current_map_level == 6:
                        self.network.save()
                        print("*****************")
                        print("Training is done!")
                        print("*****************")
                        return True
            

            self.network.writer.add_scalar('Average_step_Reward', average_reward, global_step=self.network.num_training)
            self.network.writer.add_scalar('episode_Reward', episode_reward, global_step=self.network.num_training)
            #self.network.writer.add_scalar('goal_reach_dist', self.env.goal_reach_dist, global_step=self.network.num_training)
            self.network.writer.add_scalar('Collision', collision, global_step=self.network.num_training)
            #self.network.writer.add_scalar('Done', done_bool, global_step=self.network.num_training)
            self.network.writer.add_scalar('Success_Rate', success_rate, global_step=self.network.num_training)
            self.network.writer.add_scalar('current_map_level', self.env.current_map_level, global_step=self.network.num_training)
            episode_reward      = 0

if __name__ == "__main__":
    trainer = Velodyne_Trainer(robot_num=1)
    trainer.train(robot_id=0)