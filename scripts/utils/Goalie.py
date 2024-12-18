from agent.Base_Agent import Base_Agent as Agent
from math_ops.Math_Ops import Math_Ops as M
from scripts.commons.Script import Script
import numpy as np
import random
import math

'''
Objective:
----------
Demonstrate Goalkeeper Training
'''
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
    furthest_radius = math.sqrt((furthest_point_left[0] - intersection_point[0])**2 + 
                                (furthest_point_left[1] - intersection_point[1])**2)
    closest_radius = math.sqrt((closest_point_left[0] - intersection_point[0])**2 + 
                               (closest_point_left[1] - intersection_point[1])**2)

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
    
    return x, y # , rotation

class Goalie():
    def __init__(self, script: Script) -> None:
        self.script = script

    def execute(self):

        a = self.script.args          
        player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)  # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name
        player.path_manager.draw_options(enable_obstacles=True, enable_path=True)  # Enable drawings of obstacles and path to ball
        behavior = player.behavior
        w = player.world
        r = w.robot

        print("\nThe simulation will shoot balls towards the goal randomly")
        print("Observe the ball's trajectory and goalkeeper's actions")
        print("Press ctrl+c to stop\n")

        # Set initial positions
        player.scom.unofficial_set_play_mode("PlayOn")

        # Define goal line parameters
        out_of_bounds_x = -15  # x-coordinate 
        game_time_limit = 8.0  # Maximum game time before restarting

        while True:
            # Reset game time to 0 for each iteration
            player.scom.unofficial_set_game_time(0)
            player.scom.unofficial_beam((-14, 0, r.beam_height), 0)
            behavior.execute_to_completion("Zero_Bent_Knees")

            # Generate ball position and velocity
            ball_position = random_longshot()  # x, y position
            orientation = calculate_orientation_towards_goal(ball_position)  # Angle toward goal
            ball_velocity = (
                math.cos(math.radians(orientation)) * random.uniform(10, 18),  # Increased velocity range
                math.sin(math.radians(orientation)) * random.uniform(10, 18),
                random.uniform(0.5, 8)  # Slight elevation
            )

            # Spawn the ball
            print(f"Spawning ball at {ball_position} with velocity {ball_velocity}")
            player.scom.unofficial_move_ball((*ball_position, 0.042), ball_velocity)

            # Ensure the ball is spawned
            player.scom.commit_and_send()
            player.scom.receive()

            # Inner loop to monitor the ball
            position_history = []  # Keep track of ball positions
            history_limit = 5  # Number of steps to track
            movement_threshold = 0.1  # Minimum movement distance to consider the ball moving
            stuck_counter = 0  # Count consecutive iterations with zero displacement
            stuck_limit = 30  # Maximum allowed iterations with no displacement before reset

            while True:
                # Update ball position and game time
                b = w.ball_abs_pos  # Ball absolute position (x, y, z)
                game_time = w.time_game  # Get current game time

                # # Debug position and game time
                # print(f"Ball position: {b}, Game time: {game_time}")

                # Update position history
                position_history.append(b[:2])  # Track only x, y for movement
                if len(position_history) > history_limit:
                    position_history.pop(0)  # Maintain fixed history size

                # Condition 1: Game time exceeds limit
                if game_time >= game_time_limit:
                    print("Game time limit reached!")
                    break

                # Condition 2: Ball crosses the goal line (within y-boundaries of goal)
                if b[0] <= out_of_bounds_x and -1 <= b[1] <= 1:
                    print("Ball crossed the goal line!")
                    break

                # Condition 3: Ball crosses the x-boundary (out of bounds)
                if b[0] <= out_of_bounds_x:
                    print("Ball out of bounds!")
                    break

                # Compute total displacement over the time window
                if len(position_history) == history_limit:
                    total_displacement = np.linalg.norm(np.array(position_history[-1]) - np.array(position_history[0]))
                    print(f"Ball total displacement over last {history_limit} steps: {total_displacement}")

                    if total_displacement < movement_threshold:
                        stuck_counter += 1
                    else:
                        stuck_counter = 0  # Reset counter if movement is detected

                    # Condition 4: Ball hasn't moved significantly for too long
                    if stuck_counter >= stuck_limit:
                        print("Ball appears stuck! Resetting loop.")
                        break

                # Update world state
                player.scom.commit_and_send(r.get_command()) 
                player.scom.receive()
