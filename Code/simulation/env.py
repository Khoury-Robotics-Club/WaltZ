import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

GUIEnv = True ## Set to false on cloud env
dt = 0.01 ## This the delta time for each step in the simulation

## Simple environment.
class ENV:
    def __init__(self, GUIEnv=GUIEnv):
        self.stepsCount = 0
        self.GUIEnv = GUIEnv
        self.init()

    def init(self):
        ## Set the env for the robot to interact with
        self.position_history = []
        self.velocity_history = []
        self.angular_velocity_history = []
        if self.GUIEnv:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        self.plane = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1]) ## [X,Y,Z] and quaternion
        self.robot = p.loadURDF("../urdf/robot.urdf", [0, 0, 0.25], [0, 0, 0, 1]) # useFixedBase = True, set this for robot arm
        self.rightFoot = 5  # Specify the link index you are interested in
        self.leftFoot = 2  # Specify the link index you are interested in
        self.prevObs = self.getObservation()
        self.score = 0
        self.gamma = 0.9
        
    def reset(self, GUIEnable=False):
        p.disconnect()
        self.stepsCount = 0
        self.GUIEnv = GUIEnable
        self.init()

    def vibrate_plane(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, strength=0.5):
        # Ensure strength is between 0 and 1
        strength = np.clip(strength, 0, 1)

        # Get current position and orientation of the plane
        current_position, current_orientation = p.getBasePositionAndOrientation(self.plane)

        # Calculate new position and orientation
        new_position = [
            current_position[0] + strength * (x - current_position[0]),
            current_position[1] + strength * (y - current_position[1]),
            current_position[2] + strength * (z - current_position[2])
        ]

        new_orientation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()

        # Set new position and orientation of the plane
        p.resetBasePositionAndOrientation(self.plane, new_position, new_orientation)

    def getObservation(self):
        # Get the position and orientation of the robot
        position, orientation = p.getBasePositionAndOrientation(self.robot)
        
        # Convert orientation from quaternion to roll, pitch, yaw
        rotation = R.from_quat(orientation)
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=True)
        
        _, orientation = p.getBasePositionAndOrientation(self.plane)
        rollP, pitchP, yawP = rotation.as_euler('xyz', degrees=True)

        # Get the linear and angular velocity of the robot
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot)
        
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
        contact_points = p.getContactPoints(self.robot, self.plane, linkIndexA=self.leftFoot)
        leftContact = len(contact_points) > 0

        contact_points = p.getContactPoints(self.robot, self.plane, linkIndexA=self.rightFoot)
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

    ## Actions = [lABS, lHip, lKnee, rABS, rHip, rKnee]
    def step(self, actions : np.ndarray, termination=lambda x, obj: False, reward=lambda x,y: 0):
        self.stepsCount += 1

        if len(actions) == 6:
            for idx, action in enumerate(actions):
                p.setJointMotorControlArray(self.robot, [idx], p.POSITION_CONTROL, [action])

        if GUIEnv:
            focusPos, _ = p.getBasePositionAndOrientation(self.robot)
            p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=focusPos)
        
        observation = self.getObservation()
        if termination(observation, self):
            r = reward(self.prevObs, observation)
            self.prevObs = observation
            self.score = r + self.gamma * self.score
            return "Terminated", r

        p.stepSimulation()
        if self.GUIEnv:
            time.sleep(dt)

        r = reward(self.prevObs, observation)
        self.prevObs = observation
        self.score = r + self.gamma * self.score
        return "", r
    
    def getCurScore(self) -> float:
        return self.score

def is_upside_down(observation):
    roll, pitch, yaw = observation["orientation"]["roll"], observation["orientation"]["pitch"], observation["orientation"]["yaw"]
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


if __name__ == "__main__":
    import random

    n = ENV()
    for step in range(10000):
        if step % 100 == 0:
            random_number1 = random.uniform(-5, 5)
            random_number2 = random.uniform(-5, 5)
            n.vibrate_plane(roll=random_number1, pitch=random_number2)
        actions = []
        for i in range(6):
            actions.append(random.uniform(-1, 1))
        
        actions = np.array(actions)
        err, reward = n.step(actions=actions, termination=lambda obs: terminateForFrame(obs, n), reward=scoreFunc)
        if err == "Terminated":
            n.reset()
