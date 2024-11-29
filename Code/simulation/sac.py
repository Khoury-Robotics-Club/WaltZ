import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import copy
import os  # Import os to handle directory creation


from env import ENV, terminateForFrame

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

GUIEnv = False  # Set to False on cloud environments
dt = 0.01  # Delta time for each simulation step

# Include your ENV class and supporting functions here
# For brevity, I'll assume the ENV class and other necessary functions are defined above this point.

# Define the reward function
def standing_still_reward(prevObservation, observation):
    # Constants
    max_upright_angle = 10  # degrees
    upright_bonus_weight = 1.0
    tipping_penalty_weight = 5.0
    stability_bonus_weight = 0.1
    jerk_penalty_weight = 0.05  # Adjusted to balance the impact

    # Extract orientations
    roll = observation["orientation"]["roll"]
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

# Define the neural network models for the actor and critic
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        hidden_layer_size = 512
        self.l1 = nn.Linear(state_dim, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.mean = nn.Linear(hidden_layer_size, action_dim)
        self.log_std = nn.Linear(hidden_layer_size, action_dim)
        self.min_log_std = -20
        self.max_log_std = 2

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        hidden_layer_size = 512
        self.l1 = nn.Linear(state_dim + action_dim, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l3 = nn.Linear(hidden_layer_size, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_layer_size)
        self.l5 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l6 = nn.Linear(hidden_layer_size, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        # Q1 forward
        x1 = torch.relu(self.l1(xu))
        x1 = torch.relu(self.l2(x1))
        x1 = self.l3(x1)
        # Q2 forward
        x2 = torch.relu(self.l4(xu))
        x2 = torch.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = torch.relu(self.l1(xu))
        x1 = torch.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# Define the SAC agent with the corrected train method
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        self.critic_target = copy.deepcopy(self.critic)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.target_entropy = -action_dim/2  # Recommended default value

        self.max_action = max_action
        self.discount = 0.999
        self.tau = 0.005

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            mean, _ = self.actor.forward(state)
            action = torch.tanh(mean) * self.max_action
            return action.detach().cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=512):
        # Sample replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - torch.exp(self.log_alpha) * next_log_prob
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.functional.mse_loss(current_Q1, target_Q) + nn.functional.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        action_new, log_prob = self.actor.sample(state)
        Q1_new, Q2_new = self.critic(state, action_new)
        Q_new = torch.min(Q1_new, Q2_new)
        actor_loss = (torch.exp(self.log_alpha) * log_prob - Q_new).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Temperature update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update of target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Return the losses
        return actor_loss.item(), critic_loss.item(), alpha_loss.item()

# Main training loop
if __name__ == "__main__":
    # Create 'results' directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    num_runs = 5
    num_episodes = 10000
    max_steps = 10000
    batch_size = 512
    start_steps = 10000  # Number of steps to collect data before training
    updates_per_step = 1
    action_dim = 6  # Number of actions
    max_action = 1.0  # Assuming action space is normalized between -1 and 1

    # Assuming the state consists of position (3), orientation (3), linear_velocity (3), angular_velocity (3)
    state_dim = 3 + 3 + 3 + 3  # Adjust based on your actual state representation

    for run in range(num_runs):
        # Set random seeds for reproducibility
        seed = run  # You can use any method to generate seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Initialize environment
        env = ENV(bc, GUIEnv=GUIEnv)
        # Initialize SAC agent
        agent = SACAgent(state_dim, action_dim, max_action)
        # Initialize replay buffer
        replay_buffer = ReplayBuffer()

        # Record returns and losses
        returns = []
        actor_losses = []
        critic_losses = []
        alpha_losses = []

        total_steps = 0

        for episode in range(num_episodes):
            env.reset(GUIEnable=GUIEnv)
            observation = env.getObservation()
            state = np.concatenate([
                observation['position'],
                [observation['orientation']['roll'], observation['orientation']['pitch'], observation['orientation']['yaw']],
                observation['linear_velocity'],
                observation['angular_velocity']
            ])
            episode_reward = 0

            for step in range(max_steps):
                if total_steps < start_steps:
                    action = np.random.uniform(-1, 1, size=action_dim)
                else:
                    action = agent.select_action(state)

                prev_observation = observation  # For reward calculation
                err, reward = env.step(
                    actions=action,
                    termination=lambda obs, env: terminateForFrame(obs, env),
                    reward=lambda prev_obs, obs: standing_still_reward(prev_obs, obs)
                )
                observation = env.getObservation()
                next_state = np.concatenate([
                    observation['position'],
                    [observation['orientation']['roll'], observation['orientation']['pitch'], observation['orientation']['yaw']],
                    observation['linear_velocity'],
                    observation['angular_velocity']
                ])
                done = err == "Terminated"
                replay_buffer.push(state, action, reward, next_state, float(done))

                state = next_state
                episode_reward += reward
                total_steps += 1

                if total_steps >= start_steps:
                    for _ in range(updates_per_step):
                        actor_loss, critic_loss, alpha_loss = agent.train(replay_buffer, batch_size)
                        # Record losses
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)
                        alpha_losses.append(alpha_loss)

                if done:
                    break

            returns.append(episode_reward)
            print(f"Run: {run+1}/{num_runs}, Episode: {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {step+1}")

        # Save returns and losses to numpy arrays
        np.save(f'results/returns_run_{seed}.npy', np.array(returns))
        np.save(f'results/actor_losses_run_{seed}.npy', np.array(actor_losses))
        np.save(f'results/critic_losses_run_{seed}.npy', np.array(critic_losses))
        np.save(f'results/alpha_losses_run_{seed}.npy', np.array(alpha_losses))

        # Optionally, save the trained model for each run
        torch.save(agent.actor.state_dict(), f"results/sac_actor_run_{seed}.pth")
        torch.save(agent.critic.state_dict(), f"results/sac_critic_run_{seed}.pth")
