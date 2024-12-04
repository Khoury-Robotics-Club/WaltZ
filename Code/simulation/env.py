import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
from scipy.spatial.transform import Rotation as R
import numpy as np
import time, os
import random

GUIEnv = False  # Set to False on cloud environments
dt = 0.01  # Delta time for each simulation step


def euler_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """
    Converts roll, pitch, yaw angles in degrees to a quaternion for PyBullet.
    The rotations are applied with respect to the global coordinate system (extrinsic rotations).

    Parameters:
    - roll_deg: Roll angle in degrees (rotation around global X-axis)
    - pitch_deg: Pitch angle in degrees (rotation around global Y-axis)
    - yaw_deg: Yaw angle in degrees (rotation around global Z-axis)

    Returns:
    - quaternion: A list [x, y, z, w] representing the quaternion
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # Convert degrees to radians
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    # Create a rotation object with extrinsic rotations about 'XYZ' axes
    # Uppercase 'XYZ' indicates extrinsic rotations (global coordinate system)
    rotation = R.from_euler('XYZ', [roll, pitch, yaw], degrees=False)

    # Get the quaternion (x, y, z, w)
    quaternion = rotation.as_quat()

    # Return the quaternion as a list [x, y, z, w]
    return quaternion.tolist()


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
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)  # Disable default GUI to enable user interaction
        else:
            self.p = self.bulletClient.BulletClient(connection_mode=pybullet.DIRECT)
        self.p.resetSimulation()
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)
        self.p.setRealTimeSimulation(0)

        self.plane = self.p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
        self.robot = self.p.loadURDF(self.urdf_path, [0, 0, 0.5], euler_to_quaternion(90, 0, 0))  # Raise initial height
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

        min_value = -0.785
        max_value = 0.785
        actions = actions * (max_value - min_value) + min_value
    
        if len(actions) == 6:
            for idx, action in enumerate(actions):
                self.p.setJointMotorControl2(bodyIndex=self.robot, jointIndex=idx, controlMode=pybullet.POSITION_CONTROL, targetPosition=action)

        if self.GUIEnv:
            focusPos, _ = self.p.getBasePositionAndOrientation(self.robot)
            self.p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-30, cameraTargetPosition=focusPos)

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
            r += 5000  # Reward for getting through 1000 steps
        self.prevObs = observation
        self.score = r + self.gamma * self.score
        return "", r

    def getCurScore(self) -> float:
        return self.score


def is_upside_down(observation):
    roll = observation["orientation"]["roll"]
    pitch = observation["orientation"]["pitch"]
    if abs(pitch) > 90 or abs(roll-90) > 90:
        return True
    return False


def has_moved_significantly(position_history, threshold=0.001):
    if len(position_history) < 20:
        return True  # Not enough data yet to determine
    deltas = np.diff(position_history, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    return np.any(distances > threshold)


def terminateForFrame(observation, env):
    if env.stepsCount > 2500:
        print("Count exceded")
        return True
    if is_upside_down(observation):
        print("Upside down")
        return True
    # if not has_moved_significantly(env.position_history):
    #     return True
    return False


def rewardForFrame(observation):
    return observation["position"][2] / 100


def standing_still_reward(prevObservation, observation):
    desired_roll = 90.0
    desired_pitch = 0.0
    max_angle_error = 20.0  # Maximum angle error for scaling the reward

    # Weights for different components of the reward
    upright_bonus_weight = 1.0
    tipping_penalty_weight = 5.0
    stability_bonus_weight = 0.1
    jerk_penalty_weight = 0.05  # Adjusted to balance the impact

    # Extract current orientations
    roll = observation["orientation"]["roll"]
    pitch = observation["orientation"]["pitch"]

    # Compute absolute errors
    roll_error = abs(roll - desired_roll)
    pitch_error = abs(pitch - desired_pitch)

    # Check if robot is within acceptable error margins
    within_tolerance = (roll_error < max_angle_error) and (pitch_error < max_angle_error)

    # Calculate upright_bonus based on how close the robot is to the desired orientation
    upright_bonus = upright_bonus_weight * (1 - (roll_error + pitch_error) / (2 * max_angle_error))
    upright_bonus = max(upright_bonus, 0)  # Ensure the bonus is non-negative

    # Penalize tipping over (if the error is too large)
    tipping_penalty = 0
    if roll_error >= 90 or pitch_error >= 90:
        tipping_penalty = tipping_penalty_weight

    # Stability bonus to encourage staying in the desired orientation over time
    if within_tolerance:
        stability_bonus = stability_bonus_weight
    else:
        stability_bonus = 0

    # Calculate jerk penalty based on changes in acceleration
    linear_acceleration = observation["linear_acceleration"]
    angular_acceleration = observation["angular_acceleration"]
    jerk = np.linalg.norm(linear_acceleration) + np.linalg.norm(angular_acceleration)
    jerk_penalty = jerk_penalty_weight * jerk

    # Total reward calculation
    reward = upright_bonus + stability_bonus - tipping_penalty - jerk_penalty

    return reward


# Test for a few 1000 steps with random actions.
if __name__ == "__main__":
    n = ENV(bc, GUIEnv=True)  # Enable GUI for testing
    for step in range(100000):
        actions = np.random.uniform(0, 1, size=6)
        err, reward = n.step(
            actions=actions,
            termination=lambda obs, env: terminateForFrame(obs, env),
            reward=standing_still_reward
        )
        obs = n.getObservation()
        print(obs["orientation"])
        if err == "Terminated":
            print("End of Episode")
            n.reset(True)  # Reset the environment with GUI enabled
