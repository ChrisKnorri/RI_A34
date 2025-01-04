from agent.Base_Agent import Base_Agent as Agent
from world.commons.Draw import Draw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep
import os, gym
import numpy as np
import random
import math

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
    1: "Dive_Left",
    2: "Dive_Right",
    3: "Fall_Front",
    4: "Get_Up",
    5: "shift",

}


def calculate_orientation_towards_goal(point):
    # Define the goal line
    goal_line_x = -15
    goal_line_y_min = -1
    goal_line_y_max = 1

    # Generate a random y-coordinate on the goal line
    random_goal_y = random.uniform(goal_line_y_min, goal_line_y_max)
    random_goal_point = (goal_line_x, random_goal_y)

    # Calculate the angle to the random point on the goal line
    dx = random_goal_point[0] - point[0]
    dy = random_goal_point[1] - point[1]
    angle_to_goal = math.degrees(math.atan2(dy, dx))

    # Ensure the angle is within [0, 360]
    if angle_to_goal < 0:
        angle_to_goal += 360

    return angle_to_goal


def random_longshot():
    # Define the constants for the problem
    intersection_point = (-16, 0, 0)  # Center of the circular area
    furthest_point_left = (-6, 10, 0)
    furthest_point_right = (-6, -10, 0)
    closest_point_left = (-10, 6, 0)
    closest_point_right = (-10, -6, 0)

    # Calculate the radii of the circles
    furthest_radius = math.sqrt((furthest_point_left[0] - intersection_point[0]) ** 2 +
                                (furthest_point_left[1] - intersection_point[1]) ** 2)
    closest_radius = math.sqrt((closest_point_left[0] - intersection_point[0]) ** 2 +
                               (closest_point_left[1] - intersection_point[1]) ** 2)

    # Calculate the angles of the bounding lines in radians
    angle_left = math.atan2(furthest_point_left[1] - intersection_point[1],
                            furthest_point_left[0] - intersection_point[0])
    angle_right = math.atan2(furthest_point_right[1] - intersection_point[1],
                             furthest_point_right[0] - intersection_point[0])

    # Ensure the angles are ordered correctly (left should be greater than right)
    if angle_left < angle_right:
        angle_left, angle_right = angle_right, angle_left

    # Generate a random radius and angle within the specified range
    radius = random.uniform(closest_radius, furthest_radius)
    angle = random.uniform(angle_right, angle_left)

    # Convert polar coordinates back to Cartesian coordinates
    x = intersection_point[0] + radius * math.cos(angle)
    y = intersection_point[1] + radius * math.sin(angle)

    # Calculate the rotation to face towards a random point on the goal line
    # rotation = calculate_orientation_towards_goal((x, y, 0))

    return x, y  # , rotation


class GoalkeeperEnv(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:


        self.robot_type = r_type

        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0  # to limit episode size

        #  action space: 0 = do nothing, 1 = dive left, 2 = dive right
        #               3 = fall 4 = get up
        self.action_space = spaces.Discrete(5)
        self.goalkeeper_status = 0
        self.ready = 1
        self.goal_conceded = False
        
        self.position_history = []  # Keep track of ball positions
        self.history_limit = 5  # Number of steps to track
        self.movement_threshold = 0.1  # Minimum movement distance to consider the ball moving
        self.stuck_counter = 0  # Count consecutive iterations with zero displacement
        self.stuck_limit = 40  # Maximum allowed iterations with no displacement before reset

        self.iterations = 0


        """
         Define observation space (state variables) Example:            NO  
          [ball_x, ball_y, ball_z, velocity_x, velocity_y,velocity_z,  #ball_direction ?, goalkeeper_x, goalkeeper_y,
           goalkeeper_status -> current keepers behaviour, ready]
        """
        self.observation_space = spaces.Box(
            low=np.array([-50, 0, 0, -30, -30, -30, -15, -10, 0, 0]),  # Min values
            high=np.array([50, 30, 30, 30, -30, 30, 10, 10, len(action_dict), 1]),  # Max values
            dtype=np.float32
        )
        self.obs = np.zeros(10, np.float32)
        self.state = None  # Initialize state
        assert np.any(self.player.world.robot.cheat_abs_pos), "Cheats are not enabled! Run_Utils.py -> Server -> Cheats"
        self.reset()
        
    def reset(self):
        '''
        Reset and stabilize the robot
        '''
        r = self.player.world.robot
        w = self.player.world

        # Reset variables
        goalkeeper_x = -14
        goalkeeper_y = 0
        self.ready = 1
        self.goalkeeper_status = 0
        self.goal_conceded = False
        self.step_counter = 0
        self.ball_initialized = False  # Add a flag to track ball initialization

        self.position_history = []  # Reset ball position history
        self.stuck_counter = 0

        # Clear world-specific histories
        if hasattr(self.player.world, 'ball_abs_pos_history'):
            self.player.world.ball_abs_pos_history.clear()
        
        # Stabilize goalkeeper position
        for _ in range(25):
            self.player.scom.unofficial_beam((goalkeeper_x, goalkeeper_y, 0.50), 0)
            self.player.behavior.execute("Zero")
            self.sync()

        self.player.scom.unofficial_beam((goalkeeper_x, goalkeeper_y, r.beam_height), 0)
        r.joints_target_speed[0] = 0.01  # Move head to trigger physics update
        self.sync()

        for _ in range(7):
            self.player.behavior.execute("Zero")
            self.sync()

        # Spawn ball if not already initialized
        if not self.ball_initialized:
            self.spawn_ball()
            self.ball_initialized = True

        self.fresh_episode = True
        
        return self.observe()

    def spawn_ball(self):
        '''
        Generate and spawn a ball with random position and velocity.
        '''
        ball_position = random_longshot()  # x, y position
        orientation = calculate_orientation_towards_goal(ball_position)  # Angle toward goal
        ball_velocity = (
            math.cos(math.radians(orientation)) * random.uniform(13.5, 16.5),
            math.sin(math.radians(orientation)) * random.uniform(13.5, 16.5),
            random.uniform(1, 5)  # Slight elevation
        )
        
        for _ in range(25):
            self.player.scom.unofficial_move_ball((*ball_position, 0.042), ball_velocity)
        self.sync()


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

    def predict_ball_y_at_x(self, target_x=-15):

        ball_vel_x, ball_vel_y = self.get_bal_vel()[:2]
        ball_pos_x, ball_pos_y = self.get_bal_pos()[:2]
        if ball_vel_x == 0:
            return ball_pos_y  # Assume the ball maintains its current y-position
        time_to_target_x = (target_x - ball_pos_x) / ball_vel_x

        predicted_pos_y = ball_pos_y + ball_vel_y * time_to_target_x
        return predicted_pos_y


    def observe(self):

        r = self.player.world.robot
        world = self.player.world

        ball_vel_x, ball_vel_y, ball_vel_z = world.get_ball_abs_vel(10)[:3]
        ball_pos_x, ball_pos_y, ball_pos_z = world.ball_abs_pos[:3]  # is it up to date ?
        keeper_pos_x, keeper_pos_y = r.loc_head_position[:2]  # is it up to date ?

        target_x = -15
        predicted_pos_y = self.predict_ball_y_at_x(target_x)
        # [ball_x, ball_y, velocity_x, velocity_y, ball_direction
        # , goalkeeper_x, goalkeeper_y goalkeeper_status]

        self.obs = [ball_pos_x, ball_pos_y, ball_pos_z, ball_vel_x, ball_vel_y, predicted_pos_y,
                    keeper_pos_x, keeper_pos_y, self.goalkeeper_status, self.ready]

        # Ensure no NaNs or infinities
        self.obs = np.nan_to_num(self.obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        return self.obs

    def get_bal_pos(self):
        return self.obs[0], self.obs[1],self.obs[2]

    def get_bal_vel(self):
        return self.obs[2], self.obs[3], self.obs[4]

    def get_keeper_pos(self):
        return self.obs[5], self.obs[6]

    def is_goal(self, b, out_of_bounds_x=-15):
        if b[0] <= out_of_bounds_x and -1 < b[1] < 1:
            return True
        return False

    def is_save(self, ball_abs_pos_history, out_of_bounds_x=-15):
        """
        Determines if the ball was saved based on its position history.

        Parameters:
        - ball_abs_pos_history: deque of absolute ball positions.
        - out_of_bounds_x: float, x-coordinate beyond which the ball is considered out of bounds.

        Returns:
        - bool: True if the ball was saved, False otherwise.
        """
        if len(ball_abs_pos_history) < 2:
            # Not enough data to determine a direction change or deflection
            return False

        # Current and previous positions
        current_pos = ball_abs_pos_history[0]
        previous_pos = ball_abs_pos_history[1]

        # Check if the ball is within the critical goal area
        if current_pos[0] > out_of_bounds_x and -1 <= current_pos[1] <= 1:
            # Compute the displacement vector
            displacement_x = current_pos[0] - previous_pos[0]
            displacement_y = current_pos[1] - previous_pos[1]

            # Check if the ball has moved away from the goal (deflection)
            # or has remained outside the goal area after interaction
            if displacement_x > 0 or (current_pos[1] < -1 or current_pos[1] > 1):
                # print(f"Save detected: Displacement (x, y): ({displacement_x}, {displacement_y})")
                return True

        return False

    def is_miss(self, b, out_of_bounds_x=-15):
        # If ball is out of field but not in goal
        if b[0] <= out_of_bounds_x and (b[1] < -1 or b[1] > 1):
            return True
        return False
    
    def step(self, action):
        # print(f"Step {self.step_counter}:")
        # print(f"Ball position: {self.player.world.ball_abs_pos}")
        w = self.player.world
        b = w.ball_abs_pos  # Ball absolute position (x, y, z)
        bh = w.ball_abs_pos_history
        snake_behaviour = False
        self.fresh_episode = True
        
        # Ensure action is an integer, handling scalar or array input
        if isinstance(action, np.ndarray):
            action = int(action)
        else:
            action = int(action)

        if action != self.goalkeeper_status and not self.ready:
            snake_behaviour = True

        if action != 0 and action != 5:
            behavior_name = action_dict[action]
            self.ready = self.player.behavior.execute(behavior_name)
            if self.ready:
                self.goalkeeper_status = 0
            else:
                self.goalkeeper_status = action
        elif action == 5:
            x_coordinate = self.get_keeper_pos()[0]
            y_coordinate = np.clip(self.predict_ball_y_at_x(int(x_coordinate))[1], -1.1, 1.1)
            self.player.behavior.execute("Walk", (x_coordinate, y_coordinate), True, 0, True, None)  # Args: target, is_target_abs, ori, is_ori_abs, distance

        self.sync()  # run simulation step
        self.step_counter += 1
        self.observe()

        reward = 0
        if action in [1, 2, 3, 4] and self.ready == 1:
            reward = 1 * 0
        elif action in [1, 2, 3, 4] and self.ready == 0 and snake_behaviour:
            reward = -0.1

        if self.fresh_episode and self.step_counter > 15:
            # print(f"Flags - Goal: {self.is_goal(b)}, Save: {self.is_save(bh)}, Miss: {self.is_miss(b)}")
            if self.is_goal(b):
                self.goal_conceded = True
                print("Goal! Current Step: ", self.step_counter)
                reward = -1  # Negative reward for conceding a goal
                self.fresh_episode = False
            elif self.is_save(bh) and not self.goal_conceded:
                print("Save! Current Step: ", self.step_counter)
                reward = 1  # Positive reward for saving
                self.fresh_episode = False
            elif self.is_miss(b) and self.ready == 1 and not self.goal_conceded:
                print("Miss! Current Step: ", self.step_counter)
                reward = 0.5  # Positive reward for stating ready and missing
                self.fresh_episode = False


        # Check if episode is done
        done = (
            (self.step_counter > 0 and self.step_counter >= MAX_STEP) or
            (self.step_counter > 15 and (self.is_goal(b) or self.is_save(bh) or self.is_miss(b)))
        )

        # print(f"Step Counter: {self.step_counter}, Max Step: {MAX_STEP}")
        # print(f"Done Condition: {done}")

        # Update ball position and game time
        b = w.ball_abs_pos  # Ball absolute position (x, y, z)
        self.iterations += 1

        # Update position history
        self.position_history.append(b[:2])  # Track only x, y for movement
        if len(self.position_history) > self.history_limit:
            self.position_history.pop(0)  # Maintain fixed history size

        # Compute total displacement over the time window
        if len(self.position_history) == self.history_limit and not done:
            total_displacement = np.linalg.norm(np.array(self.position_history[-1]) - np.array(self.position_history[0]))
            if total_displacement < self.movement_threshold:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0  # Reset counter if movement is detected

            # Condition 4: Ball hasn't moved significantly for too long
            if self.stuck_counter >= self.stuck_limit:
                print("Ball is not moving. Episode terminated.")
                print(f"Step {self.step_counter}: Done={done}, Reward={reward}")

                return self.state, 0, True, {}

        # Update state
        self.state = self.obs
        if done:
            print(f"Step {self.step_counter}: Done={done}, Reward={reward}")

        return self.state, reward, done, {}


class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):

        # --------------------------------------- Learning parameters
        n_envs = min(1, os.cpu_count())
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
