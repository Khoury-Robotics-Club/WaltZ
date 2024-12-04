import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import random
import os
from collections import deque
import multiprocessing as mp

# Set simulation parameters
GUIEnv = False  # Set to True to visualize the simulation
dt = 0.1  # Time step for the simulation (10 Hz)

from env import euler_to_quaternion, standing_still_reward

# Define the ENV class
class ENV:
    def __init__(self, urdf_path, GUIEnv=False):
        self.urdf_path = urdf_path
        self.GUIEnv = GUIEnv
        self.p = None
        self.init()

    def init(self):
        if self.GUIEnv:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.p.resetSimulation()
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)
        self.p.setTimeStep(dt)
        self.p.setRealTimeSimulation(0)

        # Load URDF files
        self.plane = self.p.loadURDF("plane.urdf", [0, 0, 0], euler_to_quaternion(0, 0, 0))
        self.robot = self.p.loadURDF(self.urdf_path, [0, 0, 0.25], euler_to_quaternion(90, 0, 0))

        self.prevObs = self.getObservation()
        self.score = 0
        self.gamma = 0.9

    def reset(self, state=None):
        self.p.resetSimulation()
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)
        self.p.setTimeStep(dt)
        self.p.setRealTimeSimulation(0)
        self.plane = self.p.loadURDF("plane.urdf", [0, 0, 0], euler_to_quaternion(0, 0, 0))
        self.robot = self.p.loadURDF(self.urdf_path, [0, 0, 0.25], euler_to_quaternion(90, 0, 0))

        if state is not None:
            position, orientation, linear_velocity, angular_velocity = state
            self.p.resetBasePositionAndOrientation(self.robot, position, orientation)
            self.p.resetBaseVelocity(self.robot, linear_velocity, angular_velocity)

        self.prevObs = self.getObservation()
        self.score = 0

    def getObservation(self):
        position, orientation = self.p.getBasePositionAndOrientation(self.robot)
        rotation = R.from_quat(orientation)
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=True)

        linear_velocity, angular_velocity = self.p.getBaseVelocity(self.robot)

        observation = {
            'position': np.array(position),
            'orientation': {'roll': roll, 'pitch': pitch, 'yaw': yaw},
            'linear_velocity': np.array(linear_velocity),
            'angular_velocity': np.array(angular_velocity)
        }
        return observation

    def step(self, actions: np.ndarray):
        if len(actions) == 6:
            for idx, action in enumerate(actions):
                self.p.setJointMotorControlArray(self.robot, [idx], pybullet.POSITION_CONTROL, [action])

        self.p.stepSimulation()

        observation = self.getObservation()
        return observation

    def close(self):
        if self.p is not None:
            self.p.disconnect()
            self.p = None

# Function to simulate an action sequence
def simulate_sequence(args):
    action_sequence, urdf_path, current_state = args
    # Create a copy of the environment
    sim_env = ENV(urdf_path=urdf_path, GUIEnv=False)
    sim_env.reset(state=current_state)

    total_reward = 0
    prev_obs = sim_env.getObservation()

    terminated = False
    for action in action_sequence:
        observation = sim_env.step(action)
        reward = standing_still_reward(prev_obs, observation)
        total_reward += reward
        prev_obs = observation

        # Check for termination condition
        roll_error = abs(observation["orientation"]["roll"] - 90)
        pitch_error = abs(observation["orientation"]["pitch"] - 0)
        if roll_error >= 90 or pitch_error >= 90:
            terminated = True
            break

    # If terminated early, penalize the total reward
    if terminated:
        total_reward -= 100  # Large penalty for falling over

    # Clean up the simulation environment
    sim_env.close()

    return total_reward

# MPC controller function
def mpc_controller(urdf_path, current_state, horizon, num_sequences, action_dim, max_action):
    # Generate random action sequences
    action_sequences = []
    for _ in range(num_sequences):
        action_sequence = np.random.uniform(-max_action, max_action, size=(horizon, action_dim))
        action_sequences.append(action_sequence)

    # Prepare arguments for multiprocessing
    args = [(action_sequence, urdf_path, current_state) for action_sequence in action_sequences]

    # Use multiprocessing pool to simulate sequences in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        rewards = pool.map(simulate_sequence, args)

    # Select the best action sequence
    best_sequence_index = np.argmax(rewards)
    best_action_sequence = action_sequences[best_sequence_index]

    return best_action_sequence

# Main execution block
if __name__ == "__main__":
    # Create 'results_MB' directory if it doesn't exist
    if not os.path.exists('results_MB'):
        os.makedirs('results_MB')

    # Compute the absolute URDF path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "../urdf/robot.urdf")
    urdf_path = os.path.normpath(urdf_path)

    # Initialize environment
    env = ENV(urdf_path=urdf_path, GUIEnv=GUIEnv)

    # Parameters
    horizon = 5  # MPC horizon
    num_sequences = 100  # Number of action sequences
    action_dim = 6
    max_action = 1.0
    max_steps = 1000

    # Recording results
    episode_rewards = []
    total_reward = 0
    step_count = 0

    # Get initial state
    observation = env.getObservation()
    position = observation['position']
    orientation_euler = observation['orientation']
    orientation_quat = euler_to_quaternion(
        orientation_euler['roll'],
        orientation_euler['pitch'],
        orientation_euler['yaw']
    )
    linear_velocity = observation['linear_velocity']
    angular_velocity = observation['angular_velocity']

    current_state = (position, orientation_quat, linear_velocity, angular_velocity)

    prev_observation = observation

    while True:
        # Get best action sequence from MPC
        best_action_sequence = mpc_controller(urdf_path, current_state, horizon, num_sequences, action_dim, max_action)

        # Apply the first action of the best sequence
        action = best_action_sequence[0]
        observation = env.step(action)

        # Compute reward
        reward = standing_still_reward(prev_observation, observation)
        total_reward += reward
        step_count += 1

        # Update current state
        position = observation['position']
        orientation_euler = observation['orientation']
        orientation_quat = euler_to_quaternion(
            orientation_euler['roll'],
            orientation_euler['pitch'],
            orientation_euler['yaw']
        )
        linear_velocity = observation['linear_velocity']
        angular_velocity = observation['angular_velocity']

        current_state = (position, orientation_quat, linear_velocity, angular_velocity)
        prev_observation = observation

        # Check termination condition
        roll_error = abs(orientation_euler['roll'] - 90)
        pitch_error = abs(orientation_euler['pitch'] - 0)
        done = roll_error >= 90 or pitch_error >= 90
        if done or step_count >= max_steps:
            print(f"Episode terminated after {step_count} steps.")
            break

    print(f"Total Reward: {total_reward}")

    # Save results
    np.save('results_MB/episode_rewards.npy', np.array([total_reward]))

    # Close the environment
    env.close()
