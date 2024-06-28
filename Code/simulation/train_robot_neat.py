import neat
import os
import numpy as np
from env import ENV, terminateForFrame, scoreFunc, scoreFuncWithJitter
import pybullet_utils.bullet_client as bc
from math import pi

class RobotWalker:
    def __init__(self):
        self.env = ENV(bc, False)

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            self.env.reset(False)
            genome.fitness = 0
            while True:  # Run for a certain number of steps
                obs = self.env.getObservation()
                inputs = [
                    obs['orientation']['roll'] / (2 * pi),
                    obs['orientation']['pitch'] / (2 * pi),
                    obs['orientation']['yaw'] / (2 * pi),
                    obs['orientationPlane']['roll'] / (2 * pi),
                    obs['orientationPlane']['pitch'] / (2 * pi),
                    obs['orientationPlane']['yaw'] / (2 * pi),
                    1 if obs['leftLegInContact'] else 0,
                    1 if obs['rightLegInContact'] else 0,
                    obs['position'][0],
                    obs['position'][1],
                    # obs['position'][2],  # Uncomment if needed
                    obs['linear_velocity'][0],
                    obs['linear_velocity'][1],
                    obs['linear_velocity'][2],
                    obs['angular_velocity'][0],
                    obs['angular_velocity'][1],
                    obs['angular_velocity'][2],
                    obs['linear_acceleration'][0],
                    obs['linear_acceleration'][1],
                    obs['linear_acceleration'][2],
                    obs['angular_acceleration'][0],
                    obs['angular_acceleration'][1],
                    obs['angular_acceleration'][2]
                ]

                actions = net.activate(inputs)
                actions = np.array(actions)
                actions = actions * (pi / 2)

                err, reward = self.env.step(actions=actions, termination=terminateForFrame, reward=scoreFuncWithJitter)
                genome.fitness += reward
                if err == "Terminated":
                    break

    @staticmethod
    def predict_actions(observation: dict, net):
        inputs = [
            observation['orientation']['roll'] / (2 * pi),
            observation['orientation']['pitch'] / (2 * pi),
            observation['orientation']['yaw'] / (2 * pi),
            observation['orientationPlane']['roll'] / (2 * pi),
            observation['orientationPlane']['pitch'] / (2 * pi),
            observation['orientationPlane']['yaw'] / (2 * pi),
            1 if observation['leftLegInContact'] else 0,
            1 if observation['rightLegInContact'] else 0,
            observation['position'][0],
            observation['position'][1],
            # observation['position'][2],  # Uncomment if needed
            observation['linear_velocity'][0],
            observation['linear_velocity'][1],
            observation['linear_velocity'][2],
            observation['angular_velocity'][0],
            observation['angular_velocity'][1],
            observation['angular_velocity'][2],
            observation['linear_acceleration'][0],
            observation['linear_acceleration'][1],
            observation['linear_acceleration'][2],
            observation['angular_acceleration'][0],
            observation['angular_acceleration'][1],
            observation['angular_acceleration'][2]
        ]

        actions = net.activate(inputs)
        actions = np.array(actions) * (pi / 2)  # Scale the actions if necessary
        return actions

    def test(self):
        self.env.reset()
        with open('winner.pkl', 'rb') as f:
            import pickle
            winner = pickle.load(f)

            # Load the NEAT configuration
            local_dir = os.path.dirname(__file__)
            config_path = os.path.join(local_dir, 'neat-config.ini')
            config = neat.config.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_path
            )

            # Create the neural network from the winner genome
            net = neat.nn.FeedForwardNetwork.create(winner, config)

        self.env.reset(True)

        for _ in range(10000):
            obs = self.env.getObservation()
            actions = self.predict_actions(obs, net)
            err, _ = self.env.step(actions=actions)
            if err == "Terminated":
                break
            # Print or log actions if needed
            print(f"Actions: {actions}")

def run_neat(config_file):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    walker = RobotWalker()
    winner = p.run(walker.eval_genomes, n=100)  # Run for 50 generations

    with open('winner.pkl', 'wb') as output:
        import pickle
        pickle.dump(winner, output, 1)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config.ini')
    run_neat(config_path)
