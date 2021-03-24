import numpy as np

#------------------------------------------------------------------------------

class WolfeQuapp():
  dimension = 2

  minima = (np.array([-1.174,  1.477 ]),
            np.array([-0.831, -1.366 ]),
            np.array([ 1.124, -1.486 ]))
  maxima = (np.array([ 0.100,  0.050 ]),)
  saddle = (np.array([-1.013, -0.036 ]),
            np.array([ 0.093,  0.174 ]),
            np.array([-0.208, -1.407 ]))
  
  grid_min = np.array([-2.5, -2.5])
  grid_max = np.array([ 2.5,  2.5])
  extent = np.array([grid_min[0], grid_max[0], grid_min[1], grid_max[1]])

  def potential(self, x, y):
    return x**4 + y**4 - 2.0*x**2 - 4.0*y**2 + x*y + 0.3*x + 0.1*y

  def mesh(self):
    grid_x = np.linspace(self.grid_min[0], self.grid_max[0], 200)
    grid_y = np.linspace(self.grid_min[1], self.grid_max[1], 200)
    x, y = np.meshgrid(grid_x, grid_y)
    z = self.potential(x.flatten(), y.flatten())
    return z.reshape(200, 200)

  def eval(self, position):
    assert len(position) == self.dimension
    x, y = position
    potential = x**4 + y**4 - 2.0*x**2 - 4.0*y**2 + x*y + 0.3*x + 0.1*y
    force = np.zeros(self.dimension)
    force[0] = -(4.0*x**3 - 4.0*x + y + 0.3)
    force[1] = -(4.0*y**3 - 8.0*y + x + 0.1)
    return potential, force

#------------------------------------------------------------------------------

class MuellerBrown():
  dimension = 2

  A = np.array([-200.0, -100.0, -175.0, 15.0])
  a = np.array([  -1.0,   -1.0,   -6.5,  0.7])
  b = np.array([   0.0,    0.0,   11.0,  0.6])
  c = np.array([ -10.0,  -10.0,   -6.5,  0.7])
  x0 = np.array([  1.0,    0.0,   -0.5, -1.0])
  y0 = np.array([  0.0,    0.5,    1.5,  1.0])
  minima = np.array([[-0.558, 1.442], 
                     [ 0.623, 0.028], 
                     [-0.050, 0.467]])
  saddle = np.array([[-0.822, 0.624], 
                     [-0.212, 0.293]])
                     
  grid_min = np.array([-2.0, -0.5])
  grid_max = np.array([ 1.5,  2.5])
  extent = np.array([grid_min[0], grid_max[0], grid_min[1], grid_max[1]])

  potential_scale = 0.2
  A *= potential_scale
  potential_shift = 30.331916979076915
  zmin = 0.0
  zmax = 300.0 * potential_scale

  def potential(self, x, y):
    potential = 0.0
    for i in range(4):
      potential += self.A[i] * np.exp(self.a[i]*(x-self.x0[i])**2 +
                                       self.b[i]*(x-self.x0[i])*(y-self.y0[i]) +
                                       self.c[i]*(y-self.y0[i])**2)
    return potential + self.potential_shift

  def mesh(self):
    grid_x = np.linspace(self.grid_min[0], self.grid_max[0], 200)
    grid_y = np.linspace(self.grid_min[1], self.grid_max[1], 200)
    x, y = np.meshgrid(grid_x, grid_y)
    z = self.potential(x.flatten(), y.flatten())
    return z.reshape(200, 200)

  def eval(self, position):
    assert len(position) == self.dimension
    x, y = position
    potential = 0.0
    force = np.zeros(self.dimension)
    for i in range(4):
      tmp = np.exp(self.a[i]*(x-self.x0[i])**2 +
                   self.b[i]*(x-self.x0[i])*(y-self.y0[i]) +
                   self.c[i]*(y-self.y0[i])**2)
      potential += self.A[i] * tmp
      force[0] += -self.A[i] * (2.0 * self.a[i] * (x-self.x0[i]) +
                               self.b[i] * (y-self.y0[i])) * tmp
      force[1] += -self.A[i] * (self.b[i] * (x-self.x0[i]) + 2.0 * self.c[i] *
                               (y-self.y0[i])) * tmp
    return potential + self.potential_shift, force
