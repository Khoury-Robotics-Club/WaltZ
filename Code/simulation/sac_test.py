import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import torch
import torch.nn as nn
import os
from env import ENV
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GUIEnv = True  # Set to True to visualize the simulation
dt = 0.01  # Delta time for each simulation step

def terminate_condition(observation):
    roll = observation["orientation"]["roll"]
    pitch = observation["orientation"]["pitch"]
    if abs(pitch) > 90 or abs(roll) > 90:
        return True
    return False

# Define the Actor network (same as during training)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action


        hidden_size = 512
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.min_log_std = -20
        self.max_log_std = 2

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return mean, log_std

    def select_action(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.sample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        return action.detach().cpu().numpy()

if __name__ == "__main__":
    # Compute the absolute URDF path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "../urdf/robot.urdf")
    urdf_path = os.path.normpath(urdf_path)

    # Specify the path to the saved actor model
    model_path = "results/sac_actor_run_0.pth"

    # Initialize environment
    env = ENV(bc, urdf_path=urdf_path, GUIEnv=GUIEnv)

    # State and action dimensions
    state_dim = 3 + 3 + 3 + 3  # position (3) + orientation (3) + linear_velocity (3) + angular_velocity (3)
    action_dim = 6
    max_action = 1.0

    # Load the trained actor model
    actor = Actor(state_dim, action_dim, max_action).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()

    # Reset environment
    env.reset(GUIEnable=GUIEnv)
    observation = env.getObservation()
    #print(observation['position']) ## gives -> (-0.0029145325534045696, 0.011591314338147652, 0.2183925472675824)
    #p = observation['position']
    state = np.concatenate([
        observation['position'],
        [observation['orientation']['roll'], observation['orientation']['pitch'], observation['orientation']['yaw']],
        observation['linear_velocity'],
        observation['angular_velocity']
    ])

    episode_reward = 0
    step_count = 0

    while True:
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Select action
        action = actor.select_action(state_tensor)[0]  # Get the first (and only) element

        # Step the environment
        _ = env.step(action)
        observation = env.getObservation()
        # Update state
        state = np.concatenate([
            observation['position'],
            [observation['orientation']['roll'], observation['orientation']['pitch'], observation['orientation']['yaw']],
            observation['linear_velocity'],
            observation['angular_velocity']
        ])

        # Check termination condition
        done = terminate_condition(observation)
        episode_reward += 0  # You can define a reward function if needed
        step_count += 1

        if done:
            print(f"Episode terminated after {step_count} steps.")
            break

    print(f"Total Reward: {episode_reward}")
