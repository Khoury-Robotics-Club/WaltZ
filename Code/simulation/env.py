import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
from scipy.spatial.transform import Rotation as R
import numpy as np
import time, os
import random

GUIEnv = False  # Set to False on cloud environments
dt = 0.01  # Delta time for each simulation step

# Simple environment class
class ENV:
    def __init__(self, bulletClient, GUIEnv=True, urdf_path="../urdf/robot.urdf"):
        self.stepsCount = 0
        self.GUIEnv = GUIEnv
        self.bulletClient = bulletClient
        self.p = None
        self.plane_roll = 0.0  # Initialize plane roll angle
        self.plane_pitch = 0.0  # Initialize plane pitch angle
        self.plane_yaw = 0.0  # Initialize plane yaw angle
        self.urdf_path = urdf_path
        self.init()

    def init(self):
        self.position_history = []
        self.velocity_history = []
        self.angular_velocity_history = []
        if self.GUIEnv:
            self.p = self.bulletClient.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = self.bulletClient.BulletClient(connection_mode=pybullet.DIRECT)
        self.p.resetSimulation()
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)
        self.p.setRealTimeSimulation(0)

        self.plane = self.p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
        self.robot = self.p.loadURDF(self.urdf_path, [0, 0, 0.25], [0, 0, 0, 1])
        self.rightFoot = 5
        self.leftFoot = 2
        self.prevObs = self.getObservation()
        self.score = 0
        self.gamma = 0.9

        # Reset plane angles
        self.plane_roll = 0.0
        self.plane_pitch = 0.0
        self.plane_yaw = 0.0

    def __getstate__(self):
        state = self.__dict__.copy()
        state['p'] = None  # Exclude the bullet client from the state
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.init()  # Reinitialize the bullet client

    def reset(self, GUIEnable=False):
        if self.p is None:
            self.init()
            return
        self.p.disconnect()
        self.stepsCount = 0
        self.GUIEnv = GUIEnable
        self.init()

    def getObservation(self):
        # Get the position and orientation of the robot
        position, orientation = self.p.getBasePositionAndOrientation(self.robot)
        
        # Convert orientation from quaternion to roll, pitch, yaw
        rotation = R.from_quat(orientation)
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=True)
        
        # Get plane orientation
        _, plane_orientation = self.p.getBasePositionAndOrientation(self.plane)
        plane_rotation = R.from_quat(plane_orientation)
        rollP, pitchP, yawP = plane_rotation.as_euler('xyz', degrees=True)

        # Get the linear and angular velocity of the robot
        linear_velocity, angular_velocity = self.p.getBaseVelocity(self.robot)
        
        # Calculate acceleration if we have previous velocities
        if len(self.velocity_history) > 0:
            previous_linear_velocity = self.velocity_history[-1]
            previous_angular_velocity = self.angular_velocity_history[-1]
            linear_acceleration = (np.array(linear_velocity) - np.array(previous_linear_velocity)) / dt
            angular_acceleration = (np.array(angular_velocity) - np.array(previous_angular_velocity)) / dt
        else:
            linear_acceleration = [0, 0, 0]
            angular_acceleration = [0, 0, 0]

        self.velocity_history.append(linear_velocity)
        self.angular_velocity_history.append(angular_velocity)
        if len(self.velocity_history) > 20:
            self.velocity_history.pop(0)
        if len(self.angular_velocity_history) > 20:
            self.angular_velocity_history.pop(0)

        # Check if specific link is in contact with the plane
        contact_points = self.p.getContactPoints(self.robot, self.plane, linkIndexA=self.leftFoot)
        leftContact = len(contact_points) > 0

        contact_points = self.p.getContactPoints(self.robot, self.plane, linkIndexA=self.rightFoot)
        rightContact = len(contact_points) > 0

        observation = {
            'position': position,
            'orientation': {'roll': roll, 'pitch': pitch, 'yaw': yaw},
            'orientationPlane': {'roll': rollP, 'pitch': pitchP, 'yaw': yawP},
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
            'linear_acceleration': linear_acceleration,
            'angular_acceleration': angular_acceleration,
            'leftLegInContact': leftContact,
            'rightLegInContact': rightContact
        }

        self.position_history.append(position)
        if len(self.position_history) > 20:
            self.position_history.pop(0)
        
        return observation

    # Actions = [lABS, lHip, lKnee, rABS, rHip, rKnee]
    def step(self, actions: np.ndarray, termination=lambda x, obj: False, reward=lambda x, y: 0):
        self.stepsCount += 1

        if len(actions) == 6:
            for idx, action in enumerate(actions):
                self.p.setJointMotorControlArray(self.robot, [idx], pybullet.POSITION_CONTROL, [action])

        if self.GUIEnv:
            focusPos, _ = self.p.getBasePositionAndOrientation(self.robot)
            self.p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=focusPos)
        
        # Adjust plane orientation
        max_angle = 10.0  # degrees
        max_step = 0.1    # degrees

        # Random change between -0.1 and +0.1 degrees
        delta_roll = random.uniform(-max_step, max_step)
        delta_pitch = random.uniform(-max_step, max_step)

        # Update plane angles, keeping within [-10, +10] degrees
        self.plane_roll = np.clip(self.plane_roll + delta_roll, -max_angle, max_angle)
        self.plane_pitch = np.clip(self.plane_pitch + delta_pitch, -max_angle, max_angle)

        # Apply the new orientation to the plane
        new_orientation = R.from_euler('xyz', [self.plane_roll, self.plane_pitch, self.plane_yaw], degrees=True).as_quat()
        self.p.resetBasePositionAndOrientation(self.plane, [0, 0, 0], new_orientation)

        observation = self.getObservation()
        if termination(observation, self):
            r = reward(self.prevObs, observation)
            self.prevObs = observation
            self.score = r + self.gamma * self.score
            return "Terminated", r

        self.p.stepSimulation()
        if self.GUIEnv:
            time.sleep(dt)

        r = reward(self.prevObs, observation)
        if self.stepsCount > 1000:
            r += 5000 ## Reward for geting through a 1000 steps 
        self.prevObs = observation
        self.score = r + self.gamma * self.score
        return "", r
    
    def getCurScore(self) -> float:
        return self.score

def is_upside_down(observation):
    roll = observation["orientation"]["roll"]
    pitch = observation["orientation"]["pitch"]
    if abs(pitch) > 90 or abs(roll) > 90:
        return True
    return False

def has_moved_significantly(position_history, threshold=0.001):
    if len(position_history) < 20:
        return True  # Not enough data yet to determine
    deltas = np.diff(position_history, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    return np.any(distances > threshold)

def terminateForFrame(observation, env):
    if env.stepsCount > 1000000:
        return True
    if is_upside_down(observation):
        return True
    if not has_moved_significantly(env.position_history):
        return True
    return False

def rewardForFrame(observation):
    return observation["position"][2] / 100

def scoreFunc(prevPosition, position):
    # Calculate the change in position
    prevPosition = prevPosition["position"]
    position = position["position"]

    delta_x = position[0] - prevPosition[0]
    delta_y = position[1] - prevPosition[1]
    delta_z = position[2] - prevPosition[2]

    # Reward movement in the Y direction
    reward = delta_y

    # Penalize movement in the X or Z directions
    penalty = abs(delta_x) + abs(delta_z)

    # Calculate the final score
    score = reward - penalty

    return score

def scoreFuncWithJitter(prevObservation, observation):
    # Constants for the scoring function
    stable_orientation_reward = 1.0
    jitter_penalty = 0.1
    tipped_over_penalty = 2.0
    upright_reward = 2.0
    sitting_penalty = 1.0
    desired_height = 0.2  # Desired height for the robot
    height_tolerance = 0.05  # Tolerance for the desired height
    angular_velocity_penalty = 0.5
    angular_acceleration_penalty = 0.5

    # Extract relevant data from observations
    roll = observation["orientation"]["roll"]
    pitch = observation["orientation"]["pitch"]
    position = observation["position"]
    prev_roll = prevObservation["orientation"]["roll"]
    prev_pitch = prevObservation["orientation"]["pitch"]
    angular_velocity = observation["angular_velocity"]
    angular_acceleration = observation["angular_acceleration"]

    # Calculate orientation stability reward
    orientation_stability = stable_orientation_reward / (1 + abs(roll) + abs(pitch))

    # Penalize for jitter (large changes in orientation)
    orientation_change = abs(roll - prev_roll) + abs(pitch - prev_pitch)
    jitter = jitter_penalty * orientation_change

    # Penalize for being tipped over
    tipped_over = 1 if abs(pitch) > 90 or abs(roll) > 90 else 0
    tipped_penalty = tipped_over_penalty * tipped_over

    # Reward for keeping upright at a specific height
    upright = 1 if abs(position[2] - desired_height) < height_tolerance else 0
    upright_reward = upright_reward * upright

    # Penalize if the robot is sitting down (not at the desired height)
    sitting = 1 if position[2] < desired_height - height_tolerance else 0
    sitting_penalty = sitting_penalty * sitting

    # Penalize for large angular velocities
    angular_velocity_magnitude = np.linalg.norm(angular_velocity)
    angular_velocity_pen = angular_velocity_penalty * angular_velocity_magnitude

    # Penalize for large angular accelerations
    angular_acceleration_magnitude = np.linalg.norm(angular_acceleration)
    angular_acceleration_pen = angular_acceleration_penalty * angular_acceleration_magnitude

    # Calculate the total score
    score = (
        orientation_stability
        - jitter
        - tipped_penalty
        + upright_reward
        - sitting_penalty
        - angular_velocity_pen
        - angular_acceleration_pen
    )

    return score


def standing_still_reward(prevObservation, observation):
    # Constants
    max_upright_angle = 10  # degrees
    upright_bonus_weight = 1.0
    tipping_penalty_weight = 5.0
    stability_bonus_weight = 0.1
    jerk_penalty_weight = 0.05  # Adjusted to balance the impact


    # Extract orientations
    roll = observation["orientation"]["roll"] - 90
    pitch = observation["orientation"]["pitch"]
    
    # Check if robot is upright
    upright = (abs(roll) < max_upright_angle) and (abs(pitch) < max_upright_angle)
    
    # Calculate upright_bonus
    upright_bonus = upright_bonus_weight * (1 - (abs(roll) + abs(pitch)) / (2 * max_upright_angle))
    upright_bonus = max(upright_bonus, 0)  # Ensure non-negative
    
    # Penalize tipping over
    tipping_penalty = 0
    if abs(roll) >= 90 or abs(pitch) >= 90:
        tipping_penalty = tipping_penalty_weight
    
    # Stability bonus (accumulates over time if upright)
    if upright:
        stability_bonus = stability_bonus_weight
    else:
        stability_bonus = 0

    # Calculate jerk penalty based on linear and angular accelerations
    linear_acceleration = observation["linear_acceleration"]
    angular_acceleration = observation["angular_acceleration"]
    jerk = np.linalg.norm(linear_acceleration) + np.linalg.norm(angular_acceleration)
    jerk_penalty = jerk_penalty_weight * jerk

    # Total reward
    reward = upright_bonus + stability_bonus - tipping_penalty - jerk_penalty

    return reward


## Test for a few 1000setps with random actions.
if __name__ == "__main__":
    n = ENV(bc)
    for step in range(100000): 
        #actions = np.random.uniform(-1, 1, size=6)
        actions = np.array([0, 0, 0, 0, 0, 0])
        err, reward = n.step(
            actions=actions,
            termination=lambda obs, env: terminateForFrame(obs, env),
            reward=standing_still_reward
        )
        obs = n.getObservation()
        print(obs["orientation"])
        if err == "Terminated":
            pass
            #print("End")
            #n.reset(True)
