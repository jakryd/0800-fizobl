import argparse
import numpy as np
import matplotlib.pyplot as plt
from potentials import MuellerBrown

class MC_Object():
  def __init__(self, potential, temperature, step):
      self.potential = potential
      self.dimension = potential.dimension
      self.temperature = temperature
      self.step = step
      self.beta = 1.0 / temperature
      self.position = self.random_position()
      self.traj = list()

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

  def run(self, n_samples):
    self.init_potential()
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
    ax.set_title('MC on Mueller-Brown Potential')
    plt.show()

parser = argparse.ArgumentParser(usage='python %(prog)s [options]',
                                 description='Implements simple Monte Carlo scheme.')
parser.add_argument('--n_samples', type=int, help='number of samples',
                    required=True)
parser.add_argument('--temp', type=float, help='temperature', required=True)
parser.add_argument('--step_size', type=float, default=0.1, help='step size',
                    required=False)
args = parser.parse_args()

mb_potential = MuellerBrown()
engine = MC_Object(potential=mb_potential, temperature=args.temp,
                   step=args.step_size)
engine.position = [-0.558, 1.442]
engine.run(args.n_samples)
engine.plot()
