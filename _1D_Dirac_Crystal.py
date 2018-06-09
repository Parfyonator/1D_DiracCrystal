"""
Chiral representation is used:
d  || psi_1 ||   ||     0          m - E + V(x) ||   || psi_1 ||
-- ||       || = ||                             || * ||       ||
dx || psi_2 ||   || m + E - V(x)       0        ||   || psi_2 ||

V(x) = - e^2 * sum_by_impurities( Z_i / (|x-a_i| + delta) )

Change of variable:
r(x) = sum_by_impurities( Z_i * sign(x-a_i) * ln( (|x-a_i| + delta) / delta ) )

"""

import os, re, sys, datetime, time
import numpy as np
from scipy import special
from scipy.optimize import root
from recordclass import recordclass
import multiprocessing as mp
import matplotlib.pyplot as plt


class Crystal:
  """Class that represents 1D finite crystal given by impurity positions
  and their charges.
  """

  "---Some constants---"
  SPEED_OF_LIGHT = 299792458
  ELECTRIC_CONSTANT = 8.854187817e-12
  ELECTRON_MASS = 9.10938356e-31
  PLANK_CONSTANT = 6.626070040e-34
  ELEMENTARY_CHARGE = 1.6021766208e-19
  GAMMA = 5.059074323e-8

  FINE_STRUCTURE_CONSTANT = ELEMENTARY_CHARGE**2 / (2 * ELECTRIC_CONSTANT *
                                                    SPEED_OF_LIGHT *
                                                    PLANK_CONSTANT)

  # m * c / h_bar
  X_MULT = 2 * np.pi * ELECTRON_MASS * SPEED_OF_LIGHT / PLANK_CONSTANT

  # regularization parameter (given in compton length units)
  DELTA = 0.01

  def __init__(self, A, Z, Z_p):
    """Constructor.

    Args:
        A: impurity positions.
        Z: impurity charges in elementary charge units.
        Z_p: charge of particle in elementary charge units.

    """
    self.A = A
    self.Z = Z
    self.Z_p = Z_p

  @staticmethod
  def log_transform(x, scale, origin):
    """Logarithmic transform of variable x.

    Args:
        x: point of evaluation.
        scale: scaling factor of x.
        origin: point at which logarithm argument turns into 0.

    Returns:
        Transformed variable.

    """
    return -np.log(np.abs(x - origin) * scale)

  def delta_func(self, e0, e):
    """Smeared delta function.

    Args:
        e0: functions origin.
        e: energy of evaluation.

    Returns:
        Delta function given by Lorentzian function.

    """
    return self.GAMMA / (np.pi * ((e0 - e) ** 2 + self.GAMMA ** 2))

  @staticmethod
  def interpolate(y_l, x_l, y_r, x_r, x):
    """Linear interpolation.

    Args:
        y_l: function value at left point.
        x_l: left point.
        y_r: function value at right point.
        x_r: right point.
        x: point of evaluation.

    Returns:
        Linearly interpolated value of function at point x.

    """
    return y_l + (x - x_l) * (y_r - y_l) / (x_r - x_l)

  @staticmethod
  def find_neighbours(l, val):
    """Find neighbour values in the list l for a given value val.

    Args:
        l: list of values.
        val: value for which neighbours should be found.

    Returns:
        Pair of left and right neighbours.

    """
    left = 0
    right = len(l) - 1

    while left != right - 1:
      curr = (left + right) // 2
      if l[curr] > val:
        right = curr
      else:
        left = curr

    return left, right

  @staticmethod
  def WhittakerW(k: float, m: float, x: float) -> float:
    """Gives the Whittaker function W_{k,m}(x).

    Args:
        k, m: parameters.
        x : point of evaluation.

    Returns:
        Value of Whittaker function W_{k,m} function at point x.

    """
    return np.exp(-x / 2) * pow(x, m + 0.5) * special.hyperu(0.5 + m - k,
                                                             1 + 2 * m, x)

  @staticmethod
  def WhittakerW_der(k, m, x):
    """Gives x-derivative of Whittaker function.

    Args:
        k, m: parameters.
        x: point of evaluation.

    Returns:
        Value of x-derivative Whittaker function W_{k,m} function at point x.

    """
    return ((k - 0.5 * x) * Crystal.WhittakerW(k, m, x) -
            (m * m - pow(k - 0.5, 2)) * Crystal.WhittakerW(k - 1, m, x)) / x

  def V(self, x):
    """"Calculates the potential of impurities at given position x.

    Args:
        x: point of evaluation.

    Returns:
        Value of total potential of impurities in given point.

    """
    return self.Z_p * self.FINE_STRUCTURE_CONSTANT * \
           sum([z / (np.abs(x - a) + self.DELTA) for a, z in
                zip(self.A, self.Z)])

  @staticmethod
  def max_range_for_e(e, N_nodes):
    """Calculates maximum value of x to gain necessary accuracy.

    Args:
        e: energy of particle in electron rest energy units.
        N_nodes: number of desired wave function nodes.

    Returns:
        Maximum range of calculation along x-axis.

    """
    return 10 * pow(N_nodes, 0.33) / np.sqrt(1 - e * e) if e > 0 else 10

  @staticmethod
  def normalize(psi, x):
    """Normalizes wave functions such that the total probability is 1.

    Args:
        psi: vector of particle's wave function components.
        x: list of grid points.

    """
    N = len(x)
    s = sum([(psi[i][0][0] ** 2 + psi[i][1][0] ** 2) *
             (x[i + 1] - x[i]) for i in range(N - 1)])
    psi /= np.sqrt(s)

  def h(self, x, e):
    """Calculates at given point 2x2 matrix at the rhs of the equation
    given in the description.

    Args:
        x: point of evaluation.
        e: energy of particle in electron rest energy units.

    Returns:
        2x2 matrix.

    """
    res = np.zeros((2, 2))

    mult = 1 / self.dr_by_dx(x)
    V_ = self.V(x)

    res[0, 1] = (1 - e + V_) * mult
    res[1, 0] = (1 + e - V_) * mult

    return res

  def x_2_r(self, x):
    """Turn x into r.

    Args:
        x - point of evaluation.

    Returns:
        r(x).

    """
    return sum([np.abs(z) * np.sign(x - a) *
                np.log((np.abs(x - a) + self.DELTA) / self.DELTA)
                for a, z in zip(self.A, self.Z)])

  def r_2_x(self, r):
    """Turn r into x.

    Args:
        r - point of evaluation.

    Returns:
        x(r).

    """
    return root(lambda x, r: self.x_2_r(x) - r, 0, r).x[0]

  @staticmethod
  def nodes_count(l):
    """Count the number of sign change in the given list.

    Args:
        l: list of numbers.

    Returns:
        Number of sign change in the given list.

    """
    count = 0
    for i in range(1, len(l)):
      count += np.sign(l[i]) * np.sign(l[i - 1]) < 0

    return count

  @staticmethod
  def round_seconds(current_datetime):
    """Round seconds of the given datetime to integer value
    and return datetime as a string.

    Args:
        current_datetime: current date and time.

    Returns:
        Datetime separated by '_' as a string.

    """
    s = str(current_datetime.date()) + '_'
    s += str(current_datetime.time().hour) + '-'
    s += str(current_datetime.time().minute) + '-'
    s += str(int(current_datetime.time().second))
    return s

  def RK4(self, psi, x, dr, e):
    """Solving the equation given in description by the Runge-Kutta method.

    Args:
        psi: vector of particle's wave function components.
        x: grid in x variable.
        dr: grid step in r variable.
        e: energy of particle in electron rest energy units.

    """
    N = len(x)
    for i in range(2, N, 2):
      k1 = dr * np.dot(self.h(x[i - 2], e), psi[i // 2 - 1])
      k2 = dr * np.dot(self.h(x[i - 1], e), psi[i // 2 - 1] + k1 / 2)
      k3 = dr * np.dot(self.h(x[i - 1], e), psi[i // 2 - 1] + k2 / 2)
      k4 = dr * np.dot(self.h(x[i], e), psi[i // 2 - 1] + k3)
      psi[i // 2] = psi[i // 2 - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

  def dr_by_dx(self, x):
    """Calculate a derivative of r(x) at given point.

    Args:
        x - point of evaluation.

    Returns:
        r'(x).

    """
    return sum([np.abs(z) / (np.abs(x - a) + self.DELTA) \
                for a, z in zip(self.A, self.Z)])

  def calculate(self, N, N_nodes):
    """Find energy and wave function for given conditions.

    Args:
        N: number of grid points.
        N_nodes: number of desired wave function nodes.

    Returns:
        Tuple of energy, energy error, wave function vector, r-grid and x-grid.

    """
    "---Set initial bounds for energy---"
    e_bounds = Crystal.Bounds(up=1.0, down=-1.0)

    "---'shoot' until energy is converged---"
    while True:
      "--Set energy--"
      e = 0.5 * (e_bounds.up + e_bounds.down)

      "---Calculate wave function for given energy---"
      # set default number of grid points if not given
      if not N: N = 250 * len(self.A)
      psi, x, r = self.calculate_for_given_energy(e, N_nodes, N)

      "---Count nodes for upper and lower components---"
      count_1 = self.nodes_count([psi[i][0][0] for i in range(N)])
      count_2 = self.nodes_count([psi[i][1][0] for i in range(N)])

      "---Reassign energy bounds based on node counts---"
      if count_1 >= N_nodes + 1:
        if count_2 >= N_nodes:
          e_bounds.up = e
        elif count_2 == N_nodes - 1:
          e_bounds.down = e
        else:
          e_bounds.down = e
      elif count_1 == N_nodes:
        if count_2 >= N_nodes:
          e_bounds.up = e
        else:
          e_bounds.down = e
      else:
        e_bounds.down = e

      "---Break if converged---"
      if e == 0.5 * (e_bounds.up + e_bounds.down):
        break

      "---Plot results---"
      # print(e)
      # plt.plot(x, [psi[i][0][0] for i in range(N)], 'r')
      # plt.show(block=False); plt.pause(0.1); plt.clf()

    "---Normalize wave functions---"
    self.normalize(psi, x)

    return e, (e_bounds.up - e_bounds.down) / 2, psi, r, x

  def calculate_for_given_energy(self, e, N_nodes, N=None):
    """Calculate wave functions for given energy.

    Args:
        e: particle energy.
        N_nodes: number of desired wave function nodes.
        N: number of grid points.

    Returns:
        Tuple of wave function vector, x-grid and r-grid.

    """
    "---Set grid---"
    if not N: N = 250 * len(self.A)
    x_max = self.max_range_for_e(e, N_nodes) + max(self.A)
    x_min = -self.max_range_for_e(e, N_nodes) + min(self.A)
    r_max = self.x_2_r(x_max)
    r_min = self.x_2_r(x_min)
    r = np.linspace(r_min, r_max, 2 * N - 1)
    x = [self.r_2_x(rr) for rr in r]
    psi = np.zeros((N, 2, 1))

    "---Set initial conditions---"
    korr = pow(1 - e * e, 0.5)
    p = 2 * e * self.Z_p * self.FINE_STRUCTURE_CONSTANT * sum(self.Z)
    s = 2 * e * self.Z_p * self.FINE_STRUCTURE_CONSTANT * \
        sum([z * (a + self.DELTA) for a, z in zip(self.A, self.Z)]) + \
        pow(self.Z_p * self.FINE_STRUCTURE_CONSTANT * sum(self.Z), 2)
    kappa = 0.5 * p * e / korr
    mu = -0.5 * pow(1 - 4 * s, 0.5)
    psi[0] = [[0.1], [0.1]]
    if np.abs(e) <= 1:
      psi[0][1][0] = psi[0][0][0] * (2 * korr / (1 - e + self.V(x_min))) * \
                     (self.WhittakerW_der(kappa, mu, 2 * np.abs(x_min) * korr)/
                      self.WhittakerW(kappa, mu, 2 * np.abs(x_min) * korr))
    psi[0][1][0] = 0.1

    "---Solve equation for given energy---"
    self.RK4(psi, x, r[2] - r[0], e)

    return psi, x[::2], r[::2]

  def find_energy(self, num, e_bounds):
    '''Find bound state energy in given region.

    Args:
        e_bounds: energy region.
        num: number of steps in energy region [-1, 1].

    Returns:
        A list of found energies in given energy region.

    '''
    energy_list = []
    initial_e_grid = np.linspace(e_bounds.down, e_bounds.up, num + 1)

    for i in range(num):
      # set energy interval
      e_bounds_temp = Crystal.Bounds(
        down=initial_e_grid[i],
        up=initial_e_grid[i + 1]
      )

      "--Find bound state energy if wave function sign differs on boundaries--"
      while True:
        # calculate fo energy interval boundaries
        psi_1, x_1, _ = self.calculate_for_given_energy(e_bounds_temp.down, 1)
        psi_2, x_2, _ = self.calculate_for_given_energy(e_bounds_temp.up, 1)

        # check signs
        if np.sign(psi_1[-1][0][0]) * np.sign(psi_2[-1][0][0]) > 0:
          break

        e = (e_bounds_temp.down + e_bounds_temp.up) / 2
        psi, x, _ = self.calculate_for_given_energy(e, 1)

        if np.sign(psi[-1][0][0]) * np.sign(psi_1[-1][0][0]) > 0:
          e_bounds_temp.down = e
        else:
          e_bounds_temp.up = e

        if e_bounds_temp.up - e_bounds_temp.down < 1e-15:
          energy_list.append(e)
          break

    return energy_list

  "---Ancillary function to calculate and save data for LDOS---"

  def calculate_and_save(self, i, data_dir):
    """Calculate and save LDOS data for given level.

    Args:
        i: level number.
        data_dir: directory to save data to.

    """
    '---Solve equation---'
    (e, de, psi, r, x) = self.calculate(None, i)

    "---Print results---"
    print('Level: ' + str(i))
    print('    Energy = ', e)
    print('    Energy error = ', de)

    "---Save results to file---"
    # save level
    self.save_level(e, i, psi, x)
    # save data for LDOS
    with open(data_dir + '/' + str(i) + '.txt', 'w') as out:
      out.write(str(e) + '\n')
      out.write(' '.join([str(x_i) for x_i in x]) + '\n')
      out.write(
        ' '.join([str(psi_i[0][0] ** 2 + psi_i[1][0] ** 2) for psi_i in psi]))

  def data_for_LDOS(self, N_nodes, LDOS_dir=None):
    """Calculate and save data needed for LDOS plotting.

    Args:
        N_nodes: number of desired wave function nodes.
        LDOS_dir: directory for LDOS data.

    """
    # use default name if not given
    if not LDOS_dir:
      LDOS_dir = self.crystal_dir() + '/LDOS/data'

    "---Calculate and save data in parallel---"
    cpu_count = mp.cpu_count()
    for chunk in Crystal.chunks(range(1, N_nodes + 1), cpu_count):
      processes = []
      for i in chunk:
        proc = mp.Process(target=self.calculate_and_save, args=(i, LDOS_dir))
        processes.append(proc)
        proc.start()
      for proc in processes:
        proc.join()

  @staticmethod
  def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
      yield l[i:i + n]

  def plot_LDOS(self, level_rng, filenames=None):
    """Plot local density of states (LDOS).

    Args:
        filenames: list of files with data about levels.

    """
    # Tk().withdraw()
    e_grid_N = 1000
    e_lst = []
    x_lst = []
    rho_lst = []

    if not filenames:
      data_dir = self.crystal_dir() + '/LDOS/data/'
      filenames = [data_dir + f for f in os.listdir(data_dir)
                   if os.path.isfile(data_dir + f) and
                   int(f.split('.')[0]) in level_rng]
      delta = self.DELTA
    else:
      A = []
      Z = []
      for pair in re.findall('\((.*?)\)', filenames[0], re.DOTALL):
        a, z = pair[1:-1].split(',')
        A.append(float(a))
        Z.append(float(z))
      delta = float(
        re.findall('delta=(.*?)/', self.crystal_dir(), re.DOTALL)[0])

    if len(filenames) == 0:
      sys.exit()

    print("Charges: ", [(a, z) for a, z in zip(self.A, self.Z)])
    print("Delta = ", delta)
    print("\n")

    for filename in filenames:
      with open(filename, 'r') as inp:
        e = float(inp.readline())
        x = [float(s) for s in inp.readline().split()]
        rho = [float(s) for s in inp.readline().split()]

      e_lst.append(e)
      x_lst.append(x)
      rho_lst.append(rho)

    print('Finished reading.')
    max_range_idx = 0
    max_range = 0
    for i, val in enumerate(x_lst):
      if val[-1] > max_range:
        max_range = val[-1]
        max_range_idx = i
    X = x_lst[max_range_idx]  # sorted(list(set(X)))
    delta_e = max(e_lst) - min(e_lst)
    e_grid = np.linspace(min(e_lst) - 0.001 * delta_e,
                         max(e_lst) + 0.001 * delta_e, e_grid_N)

    N = len(X)
    num_of_levels = len(e_lst)
    for level in range(num_of_levels):
      new_rho = []
      for i in range(N):
        if X[i] < x_lst[level][0] or X[i] > x_lst[level][-1]:
          new_rho.append(0.0)
        elif X[i] in x_lst[level]:
          idx = x_lst[level].index(X[i])
          new_rho.append(rho_lst[level][idx])
        else:
          (left, right) = Crystal.find_neighbours(x_lst[level], X[i])
          new_rho.append(Crystal.interpolate(rho_lst[level][left],
                                             x_lst[level][left],
                                             rho_lst[level][right],
                                             x_lst[level][right], X[i]))

      rho_lst[level] = new_rho[:]
    print('Finished recalculation.')

    LDOS = np.zeros((e_grid_N, N))

    for e_i in range(e_grid_N):
      for x_i in range(N):
        for level in range(num_of_levels):
          LDOS[e_i][x_i] += rho_lst[level][x_i] * \
                            Crystal.delta_func(self, e_lst[level], e_grid[e_i])

    # log(LDOS)
    for e_i in range(e_grid_N):
      for x_i in range(N):
        LDOS[e_i][x_i] = np.log(1 + LDOS[e_i][x_i])
    print('Finished computing LDOS.')

    "---Plot LDOS as color plot---"
    # e_grid = [log_transform(e, 10, 1) for e in e_grid]
    plt.contourf(X, e_grid, LDOS, 1000)
    # plt.colorbar()
    plt.title(r'$\delta = {0} \lambdabar_c$'.format(delta),
              fontsize=30, verticalalignment='bottom')
    plt.xlabel(r'$x$', fontsize=30, labelpad=-5)
    plt.ylabel(r'$\epsilon$  ', rotation='horizontal',
               verticalalignment='bottom', horizontalalignment='right',
               fontsize=30)
    # plt.yscale('log')
    plt.show()

  "New data type"
  Bounds = recordclass('Bounds', "up down")

  def plot_wave_function(self, psi, x):
    """Plot given wave function.

    Args:
        psi: vector of particle's wave function components.
        x: grid in x variable.

    """
    N = len(x)
    plt.plot(x, [psi[i][0][0] for i in range(N)], 'r')
    plt.plot(x, [psi[i][1][0] for i in range(N)], 'b')
    plt.title(r'$\delta = {0} \lambdabar_c$'.format(self.DELTA),
              fontsize=30, verticalalignment='bottom')
    plt.xlabel(r'$x (\lambdabar_c)$', fontsize=30)
    plt.ylabel(r'$\psi$  ', rotation='horizontal',
               verticalalignment='bottom', horizontalalignment='right',
               fontsize=30)
    plt.show()

  @staticmethod
  def plot_level(path: str):
    """Plot wave function by data from file.

    Args:
        path: path of file with data.

    """
    e, n, psi, x, A, Z = Crystal.read_level(path)
    N = len(x)
    dlt = float(re.findall('delta=(.*?)/', path, re.DOTALL)[0])
    plt.plot(x, [psi[i][0][0] for i in range(N)], 'r')
    plt.plot(x, [psi[i][1][0] for i in range(N)], 'b')
    plt.title(r'$\delta = {0} \lambdabar_c, \epsilon = {1}$'.format(dlt, e),
              fontsize=30, verticalalignment='bottom')
    plt.xlabel(r'$x (\lambdabar_c)$', fontsize=30)
    plt.ylabel(r'$\psi$  ', rotation='horizontal',
               verticalalignment='bottom', horizontalalignment='right',
               fontsize=30)
    plt.show()

  def save_level(self, e, n, psi, x):
    """Save information about the energy level n.

    Args:
        e: particle energy.
        n: level number.
        psi: vector of particle's wave function components.
        x: grid in x variable.

    """
    with open(self.crystal_dir() + '/levels/data/' + str(n) + '.txt',
              'w') as out:
      out.write(str(e) + '\n')
      out.write(str(len(x)) + '\n')
      out.write(' '.join(['({},{})'.format(a, z) for a, z in
                          zip(self.A, self.Z)]) + '\n')
      for psi_i, x_i in zip(psi, x):
        out.write(
          ' '.join([str(x_i), str(psi_i[0][0]), str(psi_i[1][0])]) + '\n')

  @staticmethod
  def read_level(path: str):
    """Read information about the energy level n.

    Args:
        path: path to file with data.

    Returns:
        Tuple of energy, level number, wave function vector,x-grid,
        list of impurity positions and list of impurity charges.

    """
    n = int(path.split('/')[-1].split('.')[0])
    with open(path, 'r') as inp:
      e = float(inp.readline())
      N = int(inp.readline())

      A = []
      Z = []
      for s in inp.readline()[:-1].split():
        a, z = s[1:-1].split(',')
        A.append(float(a))
        Z.append(float(z))

      psi = []
      x = []
      for i in range(N):
        x_i, psi_1, psi_2 = [float(var) for var in inp.readline()[:-1].split()]
        psi.append([[psi_1], [psi_2]])
        x.append(x_i)

    return e, n, psi, x, A, Z

  def crystal_dir(self):
    """Return crystal directory path, where all the information is stored.

    Returns:
        Relative directory path.

    """
    return './Results' + '/alpha=' + str(self.FINE_STRUCTURE_CONSTANT) + \
           '/delta=' + str(self.DELTA) + '/' + \
           ' '.join(['({},{})'.format(a, z) for a, z in zip(self.A, self.Z)])

  @staticmethod
  def create_dir(path):
    """Create directory with given path if doesn't exist.

    Args:
        path: directory path.

    """
    if not os.path.exists(path):
      os.mkdir(path)

  def create_file_tree(self):
    """Create file tree to store information.

    Returns:
        Relative directory path.

    """
    upper_dir = './Results'
    self.create_dir('./Results')

    upper_dir += '/alpha=' + str(self.FINE_STRUCTURE_CONSTANT)
    self.create_dir(upper_dir)

    upper_dir += '/delta=' + str(self.DELTA)
    self.create_dir(upper_dir)

    upper_dir += '/' + ' '.join(['({},{})'.format(a, z) for a, z in
                                 zip(self.A, self.Z)])
    self.create_dir(upper_dir)

    self.create_dir(upper_dir + '/levels')
    self.create_dir(upper_dir + '/levels/data')
    self.create_dir(upper_dir + '/levels/plots')

    self.create_dir(upper_dir + '/LDOS')
    self.create_dir(upper_dir + '/LDOS/data')
    self.create_dir(upper_dir + '/LDOS/plots')

    return upper_dir
