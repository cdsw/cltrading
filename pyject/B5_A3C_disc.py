# START OF DESIGN
import gym
import torch as t
import torch.nn as nn
import pandas as pd
import numpy as np
from gym import spaces
import time

init_time = time.time()

# 
# graph pyplot train, plotly test
# TRAIN
# reward_vals = []
# return_vals = []
# mdd_vals = []
import matplotlib.pyplot as plt
import pickle as pkl

save_file = "./models/B5/a3crestrain-{}.pkl"

def register(total_reward, return_h, mdd):
    tz = int(time.time() * 10000)
    r = {'reward_vals' : total_reward,
          'return_vals' : return_h,
          "mdd_vals" : mdd,
          "ts": tz}
    pkl.dump(r, open(save_file.format(tz), 'ab'))
    # print(r)



MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_UNITS = 100000
MAX_UNIT_PRICE = 200000
INITIAL_ACCOUNT_BALANCE = 10000
WINDOW_SIZE = 5
DATA_SIZE = 1
MDD_REWARD = 0.5
INTERVAL = 1000
EPISODES = 300 - 12
dsource = 'trunc_data.pkl'
model_name = 'universal_model'
df = pd.read_pickle(dsource).reset_index(level=0)
MAX_STEPS = len(df) - WINDOW_SIZE - 1
TIME_STEPS_TRAIN = MAX_STEPS * EPISODES

#100-period sortino ratio
def roll_sortino(df): #t: last time ratio
  risk_free = 0 #0 percent
  returns = (df - df.shift(-1)).to_numpy()[:-1]
  return_negative_normal = returns[returns < 0]
  return_negative_std = return_negative_normal.std() if len(return_negative_normal) > 0 else 0
  sortino_roll = (returns.mean() - risk_free) / return_negative_std * np.sqrt(100) if return_negative_std > 0 else 0
  # print("srtdf", return_negative_normal, 'SSSTD', return_negative_std, 'SRLLL', sortino_roll)
  # print("SRRL", sortino_roll)
  return sortino_roll

def sortino(df, t, algorithm): #time period: m for minute, h for hour, t: truncated
  df_truncated = df.head(t)
  df_sortino = roll_sortino(df_truncated['{}_return'.format(algorithm)].tail(100))
  # print("srt", df_truncated['sortino'], df_truncated['dqn_return'])
  sortino_ = 0 if t == 0 else df_sortino
  # print(float(sortino_))
  return float(sortino_)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class BTCTradingEnvDisc(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, algorithm):
        super(BTCTradingEnvDisc, self).__init__()

        self.df = df
        self.df['{}_return'.format(algorithm)] = 0.
        self.df['mdd'] = 0.
        self.df['sortino'] = 0.

        self.algorithm = algorithm

        # Actions (0, 1, 2)
        self.action_space = spaces.Discrete(3)

        # Prices contains the values for the last [WINDOW_SIZE] prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(DATA_SIZE + 1, WINDOW_SIZE + 1), dtype=np.float16) # 1 each dimension

        self.mdd = 0
        self.mdd_last_sell = 0
        self.sortino = 0
        self.current_reward = 0
        self.total_reward = 0

    def _next_observation(self):
        # Get the BTC data points for the last WINDOW_SIZE days and scale to between 0-16
        len_frame = len(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'close'].values)
        if DATA_SIZE == 3:
          frame = np.array([
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'close'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'ewm'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'macd_histo'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
          ])
        elif DATA_SIZE == 1:
            self.df.loc[0, 'return'] = 0
            frame = np.array([
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'return'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge'),
          ])
        elif DATA_SIZE == 4:
            frame = np.array([
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'close'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'open'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'high'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'low'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
          ])

        elif DATA_SIZE == 6:
            frame = np.array([
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'close'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'open'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'high'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'low'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'ewm'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
              np.pad(self.df.loc[self.current_step: self.current_step + WINDOW_SIZE, 'macd_histo'].to_numpy(), (0, WINDOW_SIZE + 1 - len_frame), 'edge') / MAX_UNIT_PRICE,
          ])
        
        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.units_held / MAX_NUM_UNITS,
            self.cost_basis / MAX_UNIT_PRICE,
            self.total_units_sold / MAX_NUM_UNITS,
            self.total_sales_value / (MAX_NUM_UNITS * MAX_UNIT_PRICE),
        ]], axis=0)

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "close"]
        if action == 0 and self.balance > 100:
            # Buy amount % of balance in units
            total_possible = self.balance / current_price
            units_bought = total_possible * 1
            prev_cost = self.cost_basis * self.units_held
            additional_cost = units_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.units_held + units_bought)
            self.units_held += units_bought
            self.df.loc[self.current_step, 'action'] = f'buy {units_bought:5f} @ {current_price}'

            # Update new portfolio values
            self.net_worth = self.balance + self.units_held * current_price

            if self.net_worth > self.max_net_worth:
                self.max_net_worth = self.net_worth
                self.mdd_base = self.max_net_worth # reset mdd base at ath

            if self.net_worth < self.min_net_worth:
                self.min_net_worth = self.net_worth

            if self.net_worth < self.mdd_base:
                self.mdd_base = self.net_worth # mdd calculation here

            if self.units_held == 0:
                self.cost_basis = 0

            # MDD and Sortino after buying
            # when max worth goes up, then  min_aft_max resets to the ATH
            self.mdd = max(1 - self.mdd_base/self.max_net_worth, self.mdd)

            self.df.loc[self.current_step, 'mdd'] = self.mdd
            self.sortino = sortino(self.df, self.current_step, self.algorithm) if self.current_step > 100 else 0
            self.df.loc[self.current_step, 'sortino'] = self.sortino

            # Return tracking
            self.df.loc[self.current_step, '{}_return'.format(self.algorithm)] = (self.net_worth - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE
            self.df.loc[self.current_step, 'holding'] = self.units_held
            
            # 0 reward for buying
            self.current_reward = 0#.0005
            self.df.loc[self.current_step, 'reward'] = self.current_reward#.0005

        elif action == 1 and self.units_held > 0.0001:
            # Sell amount % of units held
            units_sold = self.units_held * 1
            self.balance += units_sold * current_price
            self.units_held -= units_sold
            self.total_units_sold += units_sold
            self.total_sales_value += units_sold * current_price
            
            # Update new portfolio values
            self.net_worth = self.balance + self.units_held * current_price

            if self.net_worth > self.max_net_worth:
                self.max_net_worth = self.net_worth
                self.mdd_base = self.max_net_worth # reset mdd base at ath

            if self.net_worth < self.min_net_worth:
                self.min_net_worth = self.net_worth
            
            if self.net_worth < self.mdd_base:
                self.mdd_base = self.net_worth # mdd calculation here

            if self.units_held == 0:
                self.cost_basis = 0
            
            # MDD and Sortino after buying
            self.mdd = max(1 - self.mdd_base/self.max_net_worth, self.mdd)
            self.df.loc[self.current_step, 'mdd'] = self.mdd
            self.sortino = sortino(self.df, self.current_step, self.algorithm) if self.current_step > 100 else 0
            self.df.loc[self.current_step, 'sortino'] = self.sortino

            # Get current return (correct)
            self.df.loc[self.current_step, '{}_return'.format(self.algorithm)] = (self.net_worth - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE
            self.df.loc[self.current_step, 'holding'] = self.units_held

            # Reward calculation
            delay_modifier = (self.current_step / len(self.df))

            reward_1 = units_sold * (self.df.loc[self.current_step, 'close'] - self.cost_basis) / self.cost_basis if self.cost_basis > 1 else 0
            reward_2 = MDD_REWARD * (self.mdd_last_sell - self.df.loc[self.current_step, 'mdd'])
            reward_3 = 0#self.sortino / 10
            self.current_reward = reward_1 + reward_2 + reward_3

            self.mdd_last_sell = self.df.loc[self.current_step, 'mdd']

            self.df.loc[self.current_step, 'reward'] = f'{self.current_reward:2f} || {reward_1:3f} {reward_2:3f}'

            self.df.loc[self.current_step, 'action'] = f'sell {units_sold:5f} @ {current_price}'
        
        else: # HODL
            # Update new portfolio values
            self.net_worth = self.balance + self.units_held * current_price

            if self.net_worth > self.max_net_worth:
                self.max_net_worth = self.net_worth
                self.mdd_base = self.max_net_worth # reset mdd base at ath

            if self.net_worth < self.min_net_worth:
                self.min_net_worth = self.net_worth

            if self.net_worth < self.mdd_base:
                self.mdd_base = self.net_worth # mdd calculation here

            if self.units_held == 0:
                self.cost_basis = 0

            # MDD and Sortino after buying
            self.mdd = max(1 - self.mdd_base/self.max_net_worth, self.mdd)
            self.df.loc[self.current_step, 'mdd'] = self.mdd
            self.sortino = sortino(self.df, self.current_step, self.algorithm) if self.current_step > 100 else 0
            self.df.loc[self.current_step, 'sortino'] = self.sortino

            # Get current return (correct)
            self.df.loc[self.current_step, '{}_return'.format(self.algorithm)] = (self.net_worth - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE
            self.df.loc[self.current_step, 'holding'] = self.units_held

            self.df.loc[self.current_step, 'action'] = f'hold {self.units_held:5f} @ {self.cost_basis}'

            # Reward # HODL 
            self.current_reward = self.units_held * (0.0001 * (self.net_worth - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE) - 0.0001
            self.df.loc[self.current_step, 'reward'] = self.current_reward
        self.total_reward += self.current_reward

    def step(self, action):
        # Execute one time step within the environment
        # Sell everything at the end
        end_ = len(self.df) - WINDOW_SIZE - 1
        if self.current_step == end_ - 1:
          self._take_action(1) #sell all
        else:
          self._take_action(action)
        self.current_step += 1
        # print(self.current_step)
        # if self.current_step > len(self.df.loc[:, 'close'].values) - WINDOW_SIZE - 1 and mode != 'td3':
        #     self.current_step = self.current_step - WINDOW_SIZE - 1
        obs = self._next_observation()
        done = self.current_step == end_
        if done:
            return_h = (self.net_worth - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE
            register(self.total_reward, return_h, self.mdd)
        return obs, self.current_reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.mdd_base = INITIAL_ACCOUNT_BALANCE
        self.mdd = 0
        self.mdd_last_sell = 0
        self.min_net_worth = INITIAL_ACCOUNT_BALANCE
        self.units_held = 0
        self.cost_basis = 0
        self.total_units_sold = 0
        self.total_sales_value = 0
        self.total_reward = 0

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(0, len(self.df.loc[:, 'open'].values) - WINDOW_SIZE - 1)
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Units held: {self.units_held} (Total sold: {self.total_units_sold})')
        print(f'Avg cost for held units: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth}, min: {self.min_net_worth})')
        print(f'MDD: {self.mdd} Sortino: {self.sortino})')
        print(f'Profit: {profit}')
        print()

# END OF DESIGN

# Code is heavily inspired by Morvan Zhou's code. Please check out
# his work at github.com/MorvanZhou/pytorch-A3C

import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 64)
        self.v1 = nn.Linear(*input_dims, 64)
        self.pi = nn.Linear(64, n_actions)
        self.v = nn.Linear(64, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state, calcloss=False):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)
        
        if not calcloss:
          pi = pi[0]
          v = v[0]

        if calcloss:
          # v = t.flatten(v)
          v = v[:, 0]
          
          # print('calcfw', len(pi), len(v))

        return pi, v

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)

        R = v[-1]*(1-int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states, calcloss=True)
        values = values.squeeze()
        critic_loss = (returns-values)**2

        probs = T.softmax(pi, dim=1)[:, [0]]
        dist = Categorical(probs)
        # print('mh', actions)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action


class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, test=False):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        if name == "tester":
            self.name = name
        else:
            self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = BTCTradingEnvDisc(df, 'a3c')
        self.optimizer = optimizer
        self.test = test

    def run(self):
        t_step = 1
        num_episodes = EPISODES + 1 if self.test else EPISODES
        while self.episode_idx.value < num_episodes:
            done = False
            observation = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % MAX_STEPS == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                            self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(time.time() - init_time, ':', self.name, 'episode ', self.episode_idx.value, 'reward %.4f' % score)

class Tester(mp.Process):
    def __init__(self, global_actor_critic, input_dims, n_actions, gamma, data):
        super(Tester, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.env = BTCTradingEnvDisc(data, 'a3c') #CHANGE THIS

    def run(self):
        t_step = 1
        done = False
        observation = self.env.reset()
        score = 0
        self.local_actor_critic.clear_memory()
        while not done:
            action = self.local_actor_critic.choose_action(observation)
            observation_, reward, done, info = self.env.step(action)
            score += reward
            t_step += 1
            observation = observation_
        print(time.time() - init_time, ': tester', 'episode test', 'reward %.4f' % score)
        return self.env

if __name__ == '__main__':
    lr = 1e-4
    n_actions = 3
    input_dims = [6]
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                    optim,
                    input_dims,
                    n_actions,
                    gamma=0.95,
                    lr=lr,
                    name=i,
                    global_ep_idx=global_ep,
                    ) for i in range(mp.cpu_count())]
                    # ) for i in range(2)]
    [w.start() for w in workers]
    [w.join() for w in workers]

    data_addrs = ['hourly_bull.pkl', 'hourly_bear.pkl', 'minutely_crab.pkl', 'minutely_bull.pkl']
    for addr in data_addrs:
      dat_te = pd.read_pickle(addr)
      dat_te = dat_te.reset_index(level=0)
      tester = Tester(global_actor_critic,input_dims,n_actions,gamma=0.95,data=dat_te)
      res = tester.run()
      results_ = {'data': res.df, 'title': addr}
      pkl.dump(results_, open(f'./models/B5/res-a3c-di{addr}', 'ab'))
