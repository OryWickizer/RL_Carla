import gymnasium as gym
from gymnasium.envs.registration import register
import time

from gym_carla.envs.carla_env import CarlaEnv

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

params = {
    'number_of_vehicles': 0,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 5,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town10HD_Opt',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 150,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 4.0,  # threshold for out of lane
    'desired_speed': 4,  # desired speed (m/s)
    'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
  }

# create evaluation environment
#vec_env = make_vec_env(gym.make(CarlaEnv, n_envs=1, env_kwargs=dict(params=params)))
eval_env = gym.make('carla-v0', params=params)

#load the model
#model = PPO.load('ppo-LunarLander-v2', env=eval_env)
#model = PPO.load('ppo-LunarLander-v2_1e4', env=eval_env)
#model = PPO.load('ppo-LunarLander-v2_5e4', env=eval_env)
model = A2C.load('7')
print('model loaded')

# evaluate the agent
# deterministic True gives weird behavior --> car just goes right (or left) and nothing else
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=8, deterministic=False)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")