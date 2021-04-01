import argparse
import numpy as np
import matplotlib.pyplot as plt

class MuellerBrown_Potential():
  dimension = 2
  A = np.array([-200.0, -100.0, -175.0, 15.0])
  a = np.array([-1.0, -1.0, -6.5, 0.7])
  b = np.array([ 0.0, 0.0, 11.0, 0.6 ])
  c = np.array([-10.0, -10.0, -6.5, 0.7])
  x0 = np.array([1.0, 0.0, -0.5, -1.0])
  y0 = np.array([0.0, 0.5, 1.5, 1.0])
  minima = np.array([[ -0.558, 1.442], [0.623, 0.028], [-0.050, 0.467]])
  saddles = np.array([[ -0.822,  0.624 ], [-0.212, 0.293]])
  grid_min = np.array([-1.5, -0.5])
  grid_max = np.array([1.5, 2.5])

  def __init__(self):
    self.potential_scale = 0.2

  def eval(self, pos):
    val = np.sum(self.A * np.exp(self.a * (pos[0] - self.x0) ** 2 + self.b * (pos[0] - self.x0) * (pos[1] - self.y0) + self.c * (pos[1] - self.y0) ** 2))
    val *= self.potential_scale
    return val

class MC_Object():
  def __init__(self, potential, temperature=1.0, step=0.1, idx=0):
    self.idx = idx
    self.potential = potential
    self.dimension = potential.dimension
    self.temperature = temperature
    self.temperatures = list()
    self.step = step
    self.beta = 1.0 / temperature
    self.position = self.random_position()
    self.init_potential()
    self.traj = list()
  
  def eval(self):
    return self.potential.eval(self.position)

  def set_temperature(self, temperature):
    self.temperature = temperature
    self.beta = 1.0 / temperature

  def random_position(self):
    return np.array([
      np.random.uniform(self.potential.grid_min[i], self.potential.grid_max[i])
      for i in range(self.dimension)])

  def random_displacement(self):
    position_next = np.zeros(shape=(self.dimension))
    for i in range(self.dimension):
      delta = self.step * np.random.uniform(-1.0, 1.0)
      position_next[i] = self.position[i] + delta
    return position_next

  def init_potential(self):
    self.bins = np.full(self.dimension, 200, dtype=np.int64)
    hist = np.array([np.linspace(self.potential.grid_min[i],
                                 self.potential.grid_max[i],
                                 self.bins[i]) for i in range(self.dimension)])
    self.x_grid, self.y_grid = np.meshgrid(hist[0], hist[1])
    self.pot_grid = np.zeros(self.bins)
    for i in range(self.bins[0]):
      for j in range(self.bins[1]):
        self.pot_grid[i, j] = self.potential.eval([self.x_grid[i, j], self.y_grid[i, j]])

  def acceptance_probability(self, delta_pot):
    r = np.exp(-self.beta * delta_pot)
    if r >= 1.0:
      return 1.0
    else:
      return r

  def mc_step(self):
    curr_pos = self.position
    pos = self.random_displacement()
    delta_pot = self.potential.eval(pos) - self.potential.eval(curr_pos)
    if self.acceptance_probability(delta_pot) > np.random.uniform(0.0, 1.0):
      self.position = pos
    self.traj.append(self.position)
    self.temperatures.append(self.temperature)

  def run(self, n_samples):
    for t in range(n_samples):
      self.mc_step()

  def plot(self):
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.set_xlim([self.potential.grid_min[0], self.potential.grid_max[0]])
    ax.set_ylim([self.potential.grid_min[1], self.potential.grid_max[1]])
    cf = ax.contourf(self.x_grid, self.y_grid, self.pot_grid -
                     self.pot_grid.min(),
                     vmin=0.0, vmax=50.0, levels =
                     np.linspace(0.0, 50, 100), cmap =
                     'seismic', extend = 'max')
    plt.colorbar(cf)
    traj = np.array(self.traj)
    ax.plot(traj[:,0], traj[:,1], 'k.', alpha=0.5)
    ax.set_xlabel('$x$ coordinate')
    ax.set_ylabel('$y$ coordinate')
    ax.set_title('MC on Mueller-Brown Potential: Replica {}'.format(self.idx))
    plt.show()

parser = argparse.ArgumentParser(usage='python %(prog)s [options]',
                                 description='Implements simple Monte Carlo scheme.')
parser.add_argument('--n_samples', type=int, help='number of samples',
                    required=True)
parser.add_argument('--temp', type=lambda s: [float(l) for l in s.split(',')],
                    help='temperature', required=True)
parser.add_argument('--step_size', type=float, default=0.1, help='step size',
                    required=False)
parser.add_argument('--n_replicas', type=int, help='number of replicas',
                    required=True)
parser.add_argument('--exchange', type=int, help='stride of exchange',
                    required=True)
args = parser.parse_args()

mb_potential = MuellerBrown_Potential()

particles = list()
for r in range(args.n_replicas):
  particles.append(MC_Object(potential=mb_potential, temperature=args.temp[r],
                             step=args.step_size, idx=r))

for t in range(args.n_samples):
  for i in range(args.n_replicas):
    particles[i].mc_step()

  if (t % args.exchange == 0) and (t != 0):
    i = np.random.randint(0, args.n_replicas)
    j = np.random.randint(0, args.n_replicas)
    while i == j:
      j = np.random.randint(0, args.n_replicas)

    d_beta = particles[j].beta - particles[i].beta
    d_pot = particles[j].eval() - particles[i].eval()

    if (d_beta * d_pot > 0) or (np.random.uniform(0.0, 1.0) < np.exp(d_beta * d_pot)):
      tmp = particles[i].temperature
      particles[i].set_temperature(particles[j].temperature)
      particles[j].set_temperature(tmp)

plt.style.use('seaborn-dark-palette')
plt.figure()
for i in range(args.n_replicas):
  plt.plot(particles[i].temperatures, lw=3.0)
plt.xlabel('step')
plt.ylabel('temperature')
plt.title('Exchanges in Parallel Tempering')
plt.yticks([1, 2, 3, 4, 5])
plt.xlim([0, args.n_samples])
plt.show()

#for i in range(args.n_replicas):
#  particles[i].plot()
  
temperatures = np.zeros(shape=(args.n_replicas, args.n_samples, mb_potential.dimension))
for i in range(args.n_replicas):
	replica = particles[i]
	for j in range(args.n_samples):
		ndx = int(replica.temperatures[j] - 1)
		temperatures[ndx, j] = replica.traj[j]

bins = np.full(mb_potential.dimension, 200, dtype=np.int64)
hist = np.array([np.linspace(mb_potential.grid_min[i],
                             mb_potential.grid_max[i],
                             bins[i]) for i in range(mb_potential.dimension)])
x_grid, y_grid = np.meshgrid(hist[0], hist[1])
pot_grid = np.zeros(bins)
for i in range(bins[0]):
	for j in range(bins[1]):
	   pot_grid[i, j] = mb_potential.eval([x_grid[i, j], y_grid[i, j]])

fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.set_xlim([mb_potential.grid_min[0], mb_potential.grid_max[0]])
ax.set_ylim([mb_potential.grid_min[1], mb_potential.grid_max[1]])
cf = ax.contourf(x_grid, y_grid, pot_grid - pot_grid.min(),
                     vmin=0.0, vmax=50.0, levels =
                     np.linspace(0.0, 50, 100), cmap =
                     'seismic', extend = 'max')
plt.colorbar(cf)
plt.scatter(temperatures[0,:,0], temperatures[0,:,1], c='k')
plt.show()
