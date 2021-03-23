#! /usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(usage='python %(prog)s [options]',
                                 description='Implements particle moving on a double-well potential.')
parser.add_argument('--n_samples', type=int, help='number of samples',
                    required=True)
parser.add_argument('--temp', type=float, help='temperature', required=True)
parser.add_argument('--step_size', type=float, help='step size', required=True)
args = parser.parse_args()

num_steps = args.n_samples
delta_r = args.step_size
T = args.temp
beta = 1.0 / T
initial_position = 0.5

def potential(x):
    return 0.045*x**4-x**2
#---------------------

positions = []
total_moves = 0
accepted_moves = 0

current_position = initial_position

for i in range(num_steps):
    trial_position = current_position + delta_r * np.random.uniform(-1.0,1.0)

    current_potential = potential(current_position)
    trial_potential = potential(trial_position)

    acceptance_prob = np.exp(-beta*( trial_potential - current_potential) )
    if acceptance_prob > 1.0: acceptance_prob = 1.0
    rnd = np.random.uniform(0.0,1.0)
    if acceptance_prob > rnd:
        accepted_moves += 1
        current_position = trial_position
    total_moves += 1
    positions.append(current_position)

aver_acceptance = np.float64(accepted_moves)/np.float64(total_moves)
print('average acceptance prob = {0}'.format(aver_acceptance))

positions = np.array(positions)

plt.figure()
plt.plot(positions)
plt.xlabel('step')
plt.ylabel('$x$')
plt.title('Trajectory of $x$')
plt.show()

plt.figure()
plt.hist(positions, bins=50)
plt.xlabel('$x$')
plt.ylabel('counts')
plt.title('Histogram of $x$')
plt.show()

plt.figure()
plt.plot(np.linspace(-6, 6, 100), potential(np.linspace(-6, 6, 100)))
plt.xlabel('$x$')
plt.ylabel('potential')
plt.title('Potential $0.045 x^4-x^2$')
plt.show()
