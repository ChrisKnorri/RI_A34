from agent.Base_Agent import Base_Agent as Agent
from world.commons.Draw import Draw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep
import os, gym
import numpy as np

'''
Objective:
Learn how to fall (simplest example)
----------
- class Fall: implements an OpenAI custom gym
- class Train:  implements algorithms to train a new model or test an existing model
'''

import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

MAX_STEP = 400  # 4 seconds

action_dict = {
    0: "",
    1: "Fall_Left",
    2: "Fall_Right",
    3: "Fall_Front",
    4: "Get_Up",

}


class GoalkeeperEnv(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:

        self.robot_type = r_type

        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0  # to limit episode size

        #  action space: 0 = do nothing, 1 = dive left, 2 = dive right
        #               3 = fall 4 = get up
        self.action_space = spaces.Discrete(5)
        self.goalkeeper_status = 1



        # Define observation space (state variables) Example: [ball_x, ball_y, velocity_x, velocity_y, ball_direction ?
        # , goalkeeper_x, goalkeeper_y goalkeeper_status]
        # todo - make parameters work

        self.observation_space = spaces.Box(
            low=np.array([-50, 0, -30, 10, 0, -50, -30, 0]),  # Min values
            high=np.array([50, 30, 30, 50, 2, 50, 30, 1]),  # Max values
            dtype=np.float32
        )
        self.obs = np.zeros(8, np.float32)
        self.state = None  # Initialize state
        assert np.any(self.player.world.robot.cheat_abs_pos), "Cheats are not enabled! Run_Utils.py -> Server -> Cheats"
        self.reset()

    def reset(self):
        '''
        Reset and stabilize the robot
        '''

        r = self.player.world.robot
        world = self.player.world
        # Randomize initial state
        ball_x = np.random.uniform(-50, 50)
        ball_y = np.random.uniform(0, 30)
        velocity_x = np.random.uniform(-30, 30)
        velocity_y = np.random.uniform(10, 50)
        goalkeeper_status = 1

        # to change to real position
        goalkeeper_x = 0
        goalkeeper_y = -14

        self.step_counter = 0
        self.goalkeeper_status = 1


        for _ in range(25):
            self.player.scom.unofficial_beam((goalkeeper_x, goalkeeper_y, 0.50),
                                             0)  # beam player continuously (floating above ground)
            self.player.behavior.execute("Zero")
            self.sync()

        # beam player to ground
        self.player.scom.unofficial_beam((goalkeeper_x, goalkeeper_y, r.beam_height), 0)
        r.joints_target_speed[
            0] = 0.01  # move head to trigger physics update (rcssserver3d bug when no joint is moving)
        self.sync()

        # stabilize on ground
        for _ in range(7):
            self.player.behavior.execute("Zero")
            self.sync()

        return self.observe()

    def sync(self):
        """ Run a single simulation step """
        r = self.player.world.robot
        self.player.scom.commit_and_send(r.get_command())
        self.player.scom.receive()

    def render(self, mode='human', close=False):
        return

    def close(self):
        Draw.clear_all()
        self.player.terminate()

    def predict_ball_y_at_x(self, ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y, target_x):
        # Calculate the time it takes for the ball to reach x = -15
        time_to_target_x = (target_x - ball_pos_x) / ball_vel_x

        # Calculate the predicted y position at that time
        predicted_pos_y = ball_pos_y + ball_vel_y * time_to_target_x

        return predicted_pos_y

    def observe(self):

        r = self.player.world.robot
        world = self.player.world

        ball_vel_x, ball_vel_y = world.get_ball_abs_vel(10)[:2]
        ball_pos_x, ball_pos_y = world.ball_abs_pos[:2]  #is it up to date ?
        keeper_pos_x, keeper_pos_y = r.loc_head_position[:2] #is it up to date ?

        target_x = -15
        predicted_pos_y = self.predict_ball_y_at_x(ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y, target_x)

        #[ball_x, ball_y, velocity_x, velocity_y, ball_direction
        # , goalkeeper_x, goalkeeper_y goalkeeper_status]
        self.obs = [ball_pos_x,ball_pos_y,ball_vel_x,ball_vel_y, predicted_pos_y,keeper_pos_x,keeper_pos_y,self.goalkeeper_status]

        return self.obs



    def step(self, action):
        r = self.player.world.robot

        if action != 0:
            behavior_name = action_dict[action]
            self.player.behavior.execute_to_completion(behavior_name)

        print(f"Action in {self.step_counter} : {action}")

        self.sync()  # run simulation step
        self.step_counter += 1
        self.observe()

        reward = 0
        # self.obs[7] -> goalkeeper status
        if action in [1, 2, 3] and self.obs[7] == 1:
            self.goalkeeper_status = 0
            reward = 1 * 0
        elif action == 4:
            self.goalkeeper_status = 1
        elif action in [1, 2, 3] and self.obs[7] == 0:
            reward = -1 * 0
        elif self.obs[7] == 0:
            reward = -0.1 * 0

        if self.is_goal():
            reward = -5  # Negative reward for conceding a goal
        elif self.is_save():
            reward = 10  # Positive reward for saving
        if self.is_miss() and self.goalkeeper_status == 1:
            reward = 1  # Positive reward for stating ready

        # Check if episode is done
        done = self.is_goal() or self.is_save() or self.is_miss() or self.step_counter > MAX_STEP

        # Update state
        self.state = self.obs
        return self.state, reward, done, {}

    def is_goal(self):
        if self.get_bal_pos()[0] < -15 and abs(self.get_bal_pos()[1]) < 1:
            return True
        return False

    def is_save(self):
        # velocity of the ball in x direction is negative or zero
        return self.get_bal_vel()[0] <= 0

    def is_miss(self):
        # if ball is out of feild but not in goal
        if self.get_bal_pos()[0] < -15 and abs(self.get_bal_pos()[1]) > 1:
            return True
        return False

    def get_bal_pos(self):
        return self.obs[0], self.obs[1]

    def get_bal_vel(self):
        return self.obs[2], self.obs[3]

    def get_keeper_pos(self):
        return self.obs[5], self.obs[6]


class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):

        # --------------------------------------- Learning parameters
        n_envs = min(4, os.cpu_count())
        n_steps_per_env = 128  # RolloutBuffer is of size (n_steps_per_env * n_envs) (*RV: >=2048)
        minibatch_size = 64  # should be a factor of (n_steps_per_env * n_envs)
        total_steps = 50000  # (*RV: >=10M)
        learning_rate = 30e-4  # (*RV: 3e-4)
        # *RV -> Recommended value for more complex environments
        folder_name = f'Keeper_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        print("Model path:", model_path)

        # --------------------------------------- Run algorithm
        def init_env(i_env):
            def thunk():
                return GoalkeeperEnv(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type,
                                     False)

            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)  # include 1 extra server for testing

        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        eval_env = SubprocVecEnv([init_env(n_envs)])

        try:
            if "model_file" in args:  # retrain
                model = PPO.load(args["model_file"], env=env, n_envs=n_envs, n_steps=n_steps_per_env,
                                 batch_size=minibatch_size, learning_rate=learning_rate)
            else:  # train new model
                model = PPO("MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=minibatch_size,
                            learning_rate=learning_rate)

            model_path = self.learn_model(model, total_steps, model_path, eval_env=eval_env,
                                          eval_freq=n_steps_per_env * 10, save_freq=n_steps_per_env * 20,
                                          backup_env_file=__file__)
        except KeyboardInterrupt:
            sleep(1)  # wait for child processes
            print("\nctrl+c pressed, aborting...\n")
            servers.kill()
            return

        env.close()
        eval_env.close()
        servers.kill()

    def test(self, args):

        # Uses different server and monitor ports
        server = Server(self.server_p - 1, self.monitor_p, 1)
        env = GoalkeeperEnv(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
        model = PPO.load(args["model_file"], env=env)

        try:
            self.export_model(args["model_file"], args["model_file"] + ".pkl",
                              False)  # Export to pkl to create custom behavior
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()


'''
The learning process takes about 5 minutes.
A video with the results can be seen at:
https://imgur.com/a/KvpXS41

State space:
- Composed of all joint positions + torso height
- The number of joint positions is different for robot type 4, so the models are not interchangeable
- For this example, this problem can be avoided by using only the first 22 joints and actuators

Reward:
- The reward for falling is 1, which means that after a while every episode will have a r=1.
- What is the incetive for the robot to fall faster? Discounted return.
  In every state, the algorithm will seek short-term rewards.
- During training, the best model is saved according to the average return, which is almost always 1.
  Therefore, the last model will typically be superior for this example.

Expected evolution of episode length:
    3s|o
      |o
      | o
      |  o
      |   oo
      |     ooooo
  0.4s|          oooooooooooooooo
      |------------------------------> time


This example scales poorly with the number of CPUs because:
- It uses a small rollout buffer (n_steps_per_env * n_envs)
- The simulation workload is light
- For these reasons, the IPC overhead is significant
'''
