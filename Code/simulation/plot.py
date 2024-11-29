import numpy as np
import matplotlib.pyplot as plt
import os

def create_plots_directory():
    # Create 'plots' directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

def load_results(num_runs):
    returns_all = []
    actor_losses_all = []
    critic_losses_all = []
    alpha_losses_all = []

    for run in range(num_runs):
        seed = run
        # Load returns
        returns_path = f'results/returns_run_{seed}.npy'
        if os.path.exists(returns_path):
            returns = np.load(returns_path)
            returns_all.append(returns)
        else:
            print(f"File not found: {returns_path}")
            continue

        # Load actor losses
        actor_losses_path = f'results/actor_losses_run_{seed}.npy'
        if os.path.exists(actor_losses_path):
            actor_losses = np.load(actor_losses_path)
            actor_losses_all.append(actor_losses)
        else:
            print(f"File not found: {actor_losses_path}")
            continue

        # Load critic losses
        critic_losses_path = f'results/critic_losses_run_{seed}.npy'
        if os.path.exists(critic_losses_path):
            critic_losses = np.load(critic_losses_path)
            critic_losses_all.append(critic_losses)
        else:
            print(f"File not found: {critic_losses_path}")
            continue

        # Load alpha losses
        alpha_losses_path = f'results/alpha_losses_run_{seed}.npy'
        if os.path.exists(alpha_losses_path):
            alpha_losses = np.load(alpha_losses_path)
            alpha_losses_all.append(alpha_losses)
        else:
            print(f"File not found: {alpha_losses_path}")
            continue

    return returns_all, actor_losses_all, critic_losses_all, alpha_losses_all

def plot_returns(returns_all):
    num_runs = len(returns_all)
    num_episodes = len(returns_all[0])

    plt.figure(figsize=(10, 6))

    # Plot returns for each run
    for run, returns in enumerate(returns_all):
        plt.plot(returns, alpha=0.3, label=f'Run {run+1}')

    # Compute and plot average returns
    returns_array = np.array(returns_all)
    average_returns = np.mean(returns_array, axis=0)
    plt.plot(average_returns, color='black', linewidth=2, label='Average Return')

    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Episode Returns Over Runs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/returns_over_runs.png')
    plt.close()

def plot_losses(losses_all, loss_name):
    num_runs = len(losses_all)
    min_length = min(len(loss) for loss in losses_all)

    plt.figure(figsize=(10, 6))

    # Trim losses to the minimum length
    losses_all_trimmed = [loss[:min_length] for loss in losses_all]

    # Plot losses for each run
    for run, losses in enumerate(losses_all_trimmed):
        plt.plot(losses, alpha=0.3, label=f'Run {run+1}')

    # Compute and plot average losses
    losses_array = np.array(losses_all_trimmed)
    average_losses = np.mean(losses_array, axis=0)
    plt.plot(average_losses, color='black', linewidth=2, label=f'Average {loss_name}')

    plt.xlabel('Training Steps')
    plt.ylabel(f'{loss_name}')
    plt.title(f'{loss_name} Over Training Steps')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{loss_name.lower().replace(" ", "_")}_over_runs.png')
    plt.close()

if __name__ == "__main__":
    create_plots_directory()

    num_runs = 10  # Set the number of runs (should match the training script)

    # Load results
    returns_all, actor_losses_all, critic_losses_all, alpha_losses_all = load_results(num_runs)

    # Plot returns
    if returns_all:
        plot_returns(returns_all)
    else:
        print("No returns data available for plotting.")

    # Plot actor losses
    if actor_losses_all:
        plot_losses(actor_losses_all, 'Actor Loss')
    else:
        print("No actor losses data available for plotting.")

    # Plot critic losses
    if critic_losses_all:
        plot_losses(critic_losses_all, 'Critic Loss')
    else:
        print("No critic losses data available for plotting.")

    # Plot alpha losses
    if alpha_losses_all:
        plot_losses(alpha_losses_all, 'Alpha Loss')
    else:
        print("No alpha losses data available for plotting.")

    print("Plots have been saved in the 'plots' directory.")
