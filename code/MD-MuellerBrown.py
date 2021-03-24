import argparse
import numpy as np
import matplotlib.pyplot as plt
from potentials import MuellerBrown

def kinetic_energy(v, m=1.0):
  return 0.5 * m * (v[0]**2 + v[1]**2)

parser = argparse.ArgumentParser(usage='python %(prog)s [options]', description='Implements simple MD scheme.')
parser.add_argument('--n_samples', type=int, help='number of samples', required=True)
parser.add_argument('--dt', type=float, default=0.01, help='timestep', required=False)
parser.add_argument('--mass', type=float, default=1.0, required=False,
                    help='patricle mass')

args = parser.parse_args()

U = MuellerBrown()

position = np.zeros([args.n_samples, U.dimension])
velocity = np.zeros([args.n_samples, U.dimension])
p_energy = np.zeros(args.n_samples)
k_energy = np.zeros(args.n_samples)
times = np.arange(args.n_samples) * args.dt

position[0] = U.saddle[0]
velocity[0] = np.array([1.0, -1.0])
k_energy[0] = kinetic_energy(velocity[0])
potential, force = U.eval(position[0])
p_energy[0] = potential

dt = args.dt
m = args.mass

for i in range(0, args.n_samples-1):
  position[i+1] = position[i] + velocity[i]*dt + 0.5*(force/m)*dt**2
  p, f = U.eval(position[i+1])
  velocity[i+1] = velocity[i] + (0.5/m)*(force+f)*dt
  k_energy[i+1] = kinetic_energy(velocity[i+1], m)
  p_energy[i+1] = p
  force = f

plt.style.use('seaborn-dark-palette')
plt.figure()
plt.imshow(U.mesh(), extent=U.extent, origin='lower', vmin=U.zmin, vmax=U.zmax)
plt.colorbar(label='Potential')
plt.scatter(position[:,0], position[:,1], c='k', s=0.1)
plt.xlim([-2.0, 1.5])
plt.ylim([-0.5, 2.5])
plt.xlabel('$x$ coordinate')
plt.ylabel('$y$ coordinate')
plt.title('MD on Mueller-Brown Potential')
plt.show()

plt.figure()
plt.plot(times, position[:,0])
plt.ylim([-2.0, 1.5])
plt.xlabel('$t$ time')
plt.ylabel('$x$ coordinate')
plt.title('MD on Mueller-Brown Potential')
plt.show()

plt.figure()
plt.plot(times, position[:,1])
plt.ylim([-0.5, 2.5])
plt.xlabel('$t$ time')
plt.ylabel('$y$ coordinate')
plt.title('MD on Mueller-Brown Potential')
plt.show()
