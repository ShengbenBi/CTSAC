import argparse

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', default=0.005, type=float)                  # target smoothing coefficient
    parser.add_argument('--target_update_interval', default=2, type=int)
    parser.add_argument('--policy_update_interval', default=2, type=int)
    parser.add_argument('--gradient_steps', default=1, type=int)
    parser.add_argument('--update_interval', default=2, type=int)
    parser.add_argument('--initial_tem', default=1.0, type=float)            # Initial temperature coefficient
    parser.add_argument('--tem_decay_rate', default=0.0, type=float)         # Decay rate

    parser.add_argument('--learning_rate', default=5e-4, type=float)         
    parser.add_argument('--gamma', default=0.98, type=float)                 # discount gamma
    parser.add_argument('--capacity', default=1e5, type=int)                 # replay buffer size
    parser.add_argument('--iteration', default=5000000, type=int)            # num of games(episode)
    parser.add_argument('--batch_size', default=256, type=int)               # mini batch size
    parser.add_argument('--key_num', default=10, type=int)                   # sequence length
    parser.add_argument('--seed', default=1, type=int)
    
    parser.add_argument('--log_interval', default=300, type=int)             # Save the model once in episode
    parser.add_argument('--load', action='store_true')                       # load model
    parser.add_argument('--save', action='store_true')                       # save model
    parser.add_argument('--expl_noise', default=0.3, type=float)             # Initial exploration noise
    parser.add_argument('--expl_min', default=0.05, type=float)              # Exploration noise lower bound
    parser.add_argument('--expl_decay_steps', default=10000, type=int)       # Number of steps for decay
    parser.add_argument('--robot_num', default=1, type=int)                  # number of robots
    parser.add_argument('--random_robot_position', action='store_true')      # robot's initial position is random
    parser.add_argument('--robot_position_x', default=1.0, type=float)       # Robot initial position
    parser.add_argument('--robot_position_y', default=1.0, type=float)       # Robot initial position
    parser.add_argument('--distance_threshold', default=0.4, type=float)     # Robot repetitive position judgment threshold
    return parser
