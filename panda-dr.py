# Author: Y.B.KIM
# Date: Apr 7, 2022
"""
Domain Randomization (DR) technique used in Robot Reaching task in Reinforcement Learning (RL) setting.
The Franka Panda robot is used to reach a random placed goal point for every episodes.
    * RL environment.
    * Scene manipulation.
    * Environment resets.
    * Setting joint properties (control loop disabled, motor locked at 0 vel)
"""

import os
from os.path import dirname, join, abspath
import cv2
from pyrep import PyRep, objects
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
# from pyrep.objects.object import Object
import numpy as np
import random

import gym
from gym import spaces

from stable_baselines3 import PPO, SAC, TD3, DDPG

MODE = "train"
SCENE_FILE = join(dirname(abspath(__file__)), 'panda-dr.ttt')
POS_MIN, POS_MAX = [0.15, -0.4, 0.825], [0.625, -0.1, 0.825]
EPISODES = 15
TIMESTEPS = 200

"""
Panda_link1_responsible (visual)
Panda_link2_responsible (visual)
Panda_link3_responsible (visual)
Panda_link4_responsible (visual)
Panda_link5_responsible (visual)
Panda_link6_responsible (visual)
Panda_link7_responsible (visual)
Panda_gripper 
Panda_gripper_visual
"""


class ReacherEnv(gym.Env):

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = Panda()
        self.gripper = PandaGripper()  # Get the panda gripper from the scene

        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        
        self.target = Shape('target')
        self.dining_table = Shape('table')

        self.vision_sensor = objects.vision_sensor.VisionSensor("Vision_sensor")
        self.link1 = Shape('Panda_link1_respondable')
        self.right_finger = Shape('Panda_rightfinger_respondable')
        self.left_finger = Shape('Panda_leftfinger_respondable')
        
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(240,320,1), dtype=np.uint8)

    def get_observation(self):
        cam_image = self.vision_sensor.capture_rgb()
        gray_image = np.uint8(cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY) * 255)
        obs_image = np.expand_dims(gray_image, axis=2)
        # return {"cam_image": obs_image}
        return obs_image

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        # [Original]
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities(),
                               self.target.get_position()])
        # [Image Input]
        # return [self.]

    def reset(self):
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.dr()

        # return self._get_state()
        return self.get_observation()

    def step(self, action):

        # print("Action:", action)
        # action = list(np.random.uniform(-1.0, 1.0, size=(7)))

        
        self.agent.set_joint_target_velocities(action)  # Execute action on arm -> ERROR
        
        
        # while not self.gripper.actuate(0.5, 0.4):
        self.pr.step() # Step the physics simulation
        
        ax, ay, az = self.agent_ee_tip.get_position()
        # print("(ax, ay, az): ({:.3f},{:.3f},{:.3f})".format(ax,ay,az)) 
        tx, ty, tz = self.target.get_position()
        
        # Reward is negative distance to target
        reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)

        if ax > 0.978 or ax < 0.028 or ay > 0.5867 or ay < -0.88 or az < 0.79:
            done = True 

        if abs(reward) < 0.1:
            done = True
            reward = 3 
            self.gripper.release()
            self.gripper.grasp(self.target)
        else:
            done = False
        
        # return reward, self._get_state()
        return self.get_observation(), reward, done, {"is_success": False}

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def stop(self):
        self.pr.stop()

    def dr(self):
        """ Domain Randomization (DR) """
        # Visual Randomization
        rgb_values = list(np.random.rand(3,))
        _rgb_values = list(np.random.rand(3,))
        self.dining_table.set_color(rgb_values)
        # self.target.set_color(_rgb_values)

        # Physics Randomization 
        self.dining_table.set_bullet_friction(random.uniform(0, 1))
        self.target.set_bullet_friction(random.uniform(0, 1))
        self.link1.set_mass(3.5)
        # self.link1.set_linear_damping(0.01) # ERROR
        # self.link1.set_angular_damping(0.01)
        

# class Agent(object):

#     def act(self, state):
#         """ Random Policy """
#         # del state
#         return list(np.random.uniform(-1.0, 1.0, size=(7)))

#     def learn(self, replay_buffer):
#         del replay_buffer
#         pass


env = ReacherEnv()
# agent = Agent()
replay_buffer = []

if MODE == "train":
    print("[INFO] Train Started ...")
    model = PPO("CnnPolicy", env, verbose=1) # passed
    # model = TD3("CnnPolicy", env, verbose=1, buffer_size=100)
    model.learn(total_timesteps=5000) # total_timesteps가 학습 길이에 절대적 영향을 미침. 이때는 reward no 중요.
    print("[INFO] Model Saved ...")
    model.save("ppo_model/ppo_dr")

    env.shutdown()

# del model

elif MODE == "test":
    model_path = "sac_model/sac_dr"
    model = SAC.load(model_path)
    print("[INFO] {} loaded ...".format(model_path))

    # print("[INFO] Training Started ...")

    # for e in range(EPISODES):
    state = env.reset()
    done = False
    print("[INFO] Test Started ...")
    while not done:
        
        # avg_reward = 0
        # for i in range(EPISODE_LENGTH):
            # action = model.act(state)
        done = env.gripper.actuate(0.5, velocity=0.04)
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        
        # avg_reward += reward
        # replay_buffer.append((state, action, reward, next_state))
        # state = next_state
        # model.learn(replay_buffer)
        # model.learn(total_timesteps=TIMESTEPS)
        # model.save("ppo_model/ppo_dr")
        
        # avg_reward /= EPISODE_LENGTH

        # print("[INFO] Episode {}, Average Reward: {:.3f}".format(e+1, avg_reward))

    print('[INFO] Test Done !')
    # env.shutdown()
    env.stop()