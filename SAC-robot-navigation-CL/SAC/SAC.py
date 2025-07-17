import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from parameter import get_parameters
import datetime
import os
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
min_Val = torch.tensor(1e-7).float().to(device)
current_time = datetime.datetime.now().strftime("%m%d-%H%M%S")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=src_mask)
        x = self.ln1(x + attn_output)
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim, n_blocks, n_heads, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(embed_dim, n_heads, dropout) for _ in range(n_blocks)])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.enc_dec_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        self_attn_output, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)
        x = self.ln1(x + self_attn_output)
        enc_dec_attn_output, _ = self.enc_dec_attention(x, enc_output, enc_output, attn_mask=memory_mask)
        x = self.ln2(x + enc_dec_attn_output)
        ff_output = self.ff(x)
        x = self.ln3(x + ff_output)
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim, n_blocks, n_heads, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, n_heads, dropout) for _ in range(n_blocks)])

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, input_len, embed_dim, n_blocks, n_heads, dropout):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder = Encoder(embed_dim, n_blocks, n_heads, dropout)
        self.decoder = Decoder(embed_dim, n_blocks, n_heads, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
        return dec_output

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-10, max_log_std=2, n_blocks=2, n_heads=8, embed_dim=1024, dropout=0.1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, embed_dim)
        self.transformer = Transformer(input_len=embed_dim, embed_dim=embed_dim, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.mu_head = nn.Linear(embed_dim, action_dim)
        self.log_std_head = nn.Linear(embed_dim, action_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x, eval=False, src_mask=None, tgt_mask=None, memory_mask=None):
        x = F.relu(self.fc1(x))  # batch_size, seq_len, embed_dim
        x_att = self.transformer(x, x, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)  # batch_size, seq_len, embed_dim

        if eval: 
            x = torch.cat((x, x_att), dim=2)
            x = F.relu(self.fc2(x))
            mu = self.mu_head(x)
            log_std_head = self.log_std_head(x)
            log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
            return mu, log_std_head
        else:
            x_current = x[:, -1, :]                     
            x_att = x_att[:, -1, :]                     
            # x_att = x_att.mean(dim=1)                 
            x = torch.cat((x_current, x_att), dim=1)    
            x = F.relu(self.fc2(x))
            mu = self.mu_head(x)
            log_std_head = self.log_std_head(x)
            log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
            return mu, log_std_head

class Critic(nn.Module):  # 状态值函数V（s）
    def __init__(self, state_dim, n_blocks=2, n_heads=8, embed_dim=1024, dropout=0.1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, embed_dim)
        self.transformer = Transformer(input_len=embed_dim, embed_dim=embed_dim, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.fc4 = nn.Linear(embed_dim, 1)

    def forward(self, x, src_mask=None, tgt_mask=None, memory_mask=None):
        x = F.relu(self.fc1(x))
        x_att = self.transformer(x, x, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask) 
        x = x[:, -1, :]                      
        x_att = x_att[:, -1, :]              
        # x_att = x_att.mean(dim=1)          
        x = torch.cat((x, x_att), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Q(nn.Module):  
    def __init__(self, state_dim, action_dim, n_blocks=2, n_heads=8, embed_dim=1024, dropout=0.1):
        super(Q, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, embed_dim)
        self.transformer = Transformer(input_len=embed_dim, embed_dim=embed_dim, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.fc4 = nn.Linear(embed_dim, 1)

    def forward(self, s, a, src_mask=None, tgt_mask=None, memory_mask=None):
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x_att = self.transformer(x, x, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)  
        x = x[:, -1, :]                           
        x_att = x_att[:, -1, :]                   
        # x_att = x_att.mean(dim=1)               
        x = torch.cat((x, x_att), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class SAC():
    def __init__(self, state_dim, action_dim, max_action):
        super(SAC, self).__init__()

        parser = get_parameters()
        self.args = parser.parse_args()
        # Set seeds
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

        self.state_dim        = state_dim
        self.action_dim       = action_dim
        self.max_action       = max_action

        self.policy_net       = Actor(self.state_dim, self.action_dim).to(device)
        self.value_net        = Critic(self.state_dim).to(device)
        self.Target_value_net = Critic(self.state_dim).to(device)
        self.Q_net1           = Q(self.state_dim, self.action_dim).to(device)
        self.Q_net2           = Q(self.state_dim, self.action_dim).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.args.learning_rate)
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=self.args.learning_rate)
        self.Q1_optimizer     = optim.Adam(self.Q_net1.parameters(), lr=self.args.learning_rate)
        self.Q2_optimizer     = optim.Adam(self.Q_net2.parameters(), lr=self.args.learning_rate)

        self.num_training   = 1
        self.initial_tem    = self.args.initial_tem
        self.tem_decay_rate = self.args.tem_decay_rate

        self.last_map_level = 0
        self.num_training_map_level = 0
        
        log_dir             = './runs/' + current_time
        self.writer         = SummaryWriter(log_dir=log_dir)

        # calculate the loss
        self.value_criterion = nn.MSELoss()
        self.Q1_criterion    = nn.MSELoss()
        self.Q2_criterion    = nn.MSELoss()

        # copy the weight of value_net to Target_value_net
        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, state):
        state      = torch.FloatTensor(state).to(device)
        mu, log_sigma = self.policy_net(state)
        sigma      = torch.exp(log_sigma)
        dist       = Normal(mu, sigma)
        z          = dist.sample()
        action     = torch.tanh(z).detach().cpu().numpy()  
        return action  

    def evaluate(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state, eval=True)
        batch_sigma      = torch.exp(batch_log_sigma)
        dist             = Normal(batch_mu, batch_sigma)
        noise            = Normal(0, 1)
        z                = noise.sample()
        action           = torch.tanh(batch_mu + batch_sigma*z.to(device))
        log_prob         = dist.log_prob(batch_mu + batch_sigma * z.to(device)) - torch.log(1 - action.pow(2) + min_Val)
        log_prob = log_prob.sum(dim = 2, keepdim=True) / 2  
        return action, log_prob, z, batch_mu, batch_log_sigma
    
    def update(self, replay_buffer, batch_size, current_map_level):
        for _ in range(self.args.gradient_steps):

            self.tem = self.initial_tem / (1.0 + self.tem_decay_rate * self.num_training_map_level)

            # Sample a batch from replaybuffer
            (s, a, r, s_, d) = replay_buffer.sample_batch(batch_size, self.args.keys_num)
            bn_s  = torch.Tensor(s).float().to(device)
            bn_a  = torch.Tensor(a).to(device)
            bn_r  = torch.Tensor(r).to(device)
            bn_s_ = torch.Tensor(s_).float().to(device)
            bn_d  = torch.Tensor(d).float().to(device)

            bn_r = bn_r.reshape(-1, self.args.keys_num, 1)[:, -1, :]
            bn_d = bn_d.reshape(-1, self.args.keys_num, 1)[:, -1, :]
            target_value = self.Target_value_net(bn_s_)
            next_q_value = bn_r + (1 - bn_d) * self.args.gamma * target_value

            excepted_value = self.value_net(bn_s)
            excepted_Q1    = self.Q_net1(bn_s, bn_a)
            excepted_Q2    = self.Q_net2(bn_s, bn_a)

            sample_action, log_prob, *_ = self.evaluate(bn_s)
            log_prob = log_prob[:, -1, :]
            excepted_new_Q = torch.min(self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action))
            next_value     = excepted_new_Q - self.tem * log_prob  #J_V

            # !!!Note that the actions are sampled according to the current policy,
            # instead of replay buffer. (From original paper)
            V_loss = self.value_criterion(excepted_value, next_value.detach()).mean()  # J_V

            # Dual Q net
            Q1_loss = self.Q1_criterion(excepted_Q1, next_q_value.detach()).mean() # J_Q
            Q2_loss = self.Q2_criterion(excepted_Q2, next_q_value.detach()).mean()
            
            self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss, global_step=self.num_training)

            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q1_optimizer.zero_grad()
            Q1_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            Q2_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()
            if current_map_level != self.last_map_level:
                self.num_training_map_level = 0

            if self.num_training % self.args.policy_update_interval == 0:
                self.policy_optimizer.zero_grad()
                # Recalculate pi_loss with new variables
                sample_action, log_prob, *_ = self.evaluate(bn_s.detach())
                log_prob = log_prob[:, -1, :]
                excepted_new_Q = torch.min(self.Q_net1(bn_s.detach(), sample_action), self.Q_net2(bn_s.detach(), sample_action))
                pi_loss = (self.tem*log_prob - excepted_new_Q).mean()
                self.writer.add_scalar('Loss/policy_loss', pi_loss, global_step=self.num_training)
                pi_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()

            # update target v net update
            if self.num_training % self.args.target_update_interval == 0:
                for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                    target_param.data.copy_(target_param * (1 - self.args.tau) + param * self.args.tau)

            self.num_training += 1     
            self.last_map_level = current_map_level
            self.num_training_map_level += 1 

    def save(self):

        save_dir = f'./pytorch_models/{current_time}'
        os.makedirs(save_dir, exist_ok=True)
        

        torch.save(self.policy_net.state_dict(), os.path.join(save_dir, 'policy_net.pth'))
        torch.save(self.value_net.state_dict(), os.path.join(save_dir, 'value_net.pth'))
        torch.save(self.Q_net1.state_dict(), os.path.join(save_dir, 'Q_net1.pth'))
        torch.save(self.Q_net2.state_dict(), os.path.join(save_dir, 'Q_net2.pth'))
        
        print("====================================")
        print(f"Models have been saved in {save_dir}...")
        print("====================================")

    def load(self):
        self.policy_net.load_state_dict(torch.load('./pytorch_models/policy_net.pth'))
        self.value_net.load_state_dict(torch.load ('./pytorch_models/value_net.pth'))
        self.Q_net1.load_state_dict(torch.load    ('./pytorch_models/Q_net1.pth'))
        self.Q_net2.load_state_dict(torch.load    ('./pytorch_models/Q_net2.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
