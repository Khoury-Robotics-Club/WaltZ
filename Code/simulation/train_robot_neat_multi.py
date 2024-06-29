import neat
import os
import numpy as np
from env import ENV, terminateForFrame, scoreFunc, scoreFuncWithJitter
from math import pi
import pybullet_utils.bullet_client as bc
import time
import multiprocessing as mp

def evalSingle(genome, config, c_conn=None):
    env = ENV(bc, False)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env.reset(False)
    genome.fitness = 0
    startTime = time.time()
    while True:
        obs = env.getObservation()
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

        err, reward = env.step(actions=actions, termination=terminateForFrame, reward=scoreFuncWithJitter)
        genome.fitness += reward
        if err == "Terminated":
            break
        if time.time() - startTime > 100.0:
            break
    
    if c_conn == None:
        return genome.fitness
    else:
        c_conn.send(genome.fitness)

def eval_multi(genomes, config):
    cpuCount = mp.cpu_count()
    popSize = config.pop_size
    processCount = popSize
    if cpuCount > popSize:
        processCount = cpuCount - 1

    pipes = dict()
    processes = dict()
    for idx, genome in genomes:
        p_conn, c_conn = mp.Pipe()
        pipes[idx] = ([p_conn, c_conn])
        process = mp.Process(target=evalSingle, args=(genome, config, c_conn))
        process.start()
        processes[idx] = process

    for idx, genome in genomes:
        p_conn = pipes[idx][0]
        genome.fitness = p_conn.recv()
        processes[idx].join()
        
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

    winner = p.run(eval_multi, n=100)  # Run for 50 generations

    with open('winner.pkl', 'wb') as output:
        import pickle
        pickle.dump(winner, output, 1)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config.ini')
    run_neat(config_path)
