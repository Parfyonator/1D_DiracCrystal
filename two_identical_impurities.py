from _1D_Dirac_Crystal import Crystal
import os, datetime, re
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np


class TwoSameImpurities(Crystal):
    """Crystal made of two identical impurities.
    """

    def __init__(self, a, Z_i, Z_p):
        """Constructor.

        Args:
            a: right impurity position.
            Z_i: impurity charge in elementary charge units.
            Z_p: charge of particle in elementary charge units.

        """
        super(TwoSameImpurities, self).__init__([-a, a], [Z_i, Z_i], Z_p)

    def create_file_tree(self):
        """Create file tree."""
        curr_dir = super(TwoSameImpurities, self).create_file_tree()
        curr_dir = curr_dir[:curr_dir.rfind('/')]

        Crystal.create_dir(curr_dir + '/e_vs_Zcr')
        Crystal.create_dir(curr_dir + '/e_vs_Zcr/data')
        Crystal.create_dir(curr_dir + '/e_vs_Zcr/plots')

        Crystal.create_dir(curr_dir + '/e_vs_R')
        Crystal.create_dir(curr_dir + '/e_vs_R/data')
        Crystal.create_dir(curr_dir + '/e_vs_R/plots')

        Crystal.create_dir(curr_dir + '/Zcr_vs_R')
        Crystal.create_dir(curr_dir + '/Zcr_vs_R/data')
        Crystal.create_dir(curr_dir + '/Zcr_vs_R/plots')

    @staticmethod
    def delta_dir():
        """Delta directory."""
        return './Results/alpha=' + str(Crystal.FINE_STRUCTURE_CONSTANT) + '/delta=' + str(Crystal.DELTA)

    def x_2_r(self, x):
        """Turn x into r.

        Args:
            x: point of evaluation.

        Returns:
            r(x).

        """
        R = np.abs(self.A[0])
        Z_i = self.Z[0]
        if x < -R:
            return Z_i * np.log(self.DELTA**2 / ((self.DELTA - R - x) * (R - x + self.DELTA)))
        elif x < R:
            return Z_i * np.log((x + R + self.DELTA) / (R - x + self.DELTA))
        else:
            return Z_i * np.log((self.DELTA + R + x) * (x - R + self.DELTA) / self.DELTA**2)

    def r_2_x(self, r):
        """Turn r into x.

        Args:
            r: point of evaluation.

        Returns:
            x(r).

        """
        R = np.abs(self.A[0])
        R_2_r = self.x_2_r(R)
        Z_i = self.Z[0]
        if r < -R_2_r:
            return self.DELTA - np.sqrt(self.DELTA**2 * np.exp(-r/Z_i) + R**2)
        elif r < R_2_r:
            return (R + self.DELTA) * np.tanh(r/(2*Z_i))
        else:
            return np.sqrt(self.DELTA**2 * np.exp(r/Z_i) + R**2) - self.DELTA

    @staticmethod
    def user_interface():
        """Function to get parameters from user and solve Dirac equation numerically."""
        while True:
            "---Ask user for parameters---"
            R = float(input('Enter position of impurity (in compton lenght units): '))
            Z_i, Z_p = [float(elem) for elem in \
                        input('Enter charges of impurity and charge carrier in elementary charge units: ').split()]
            N = 1 + int(input('Enter the number of grid points: '))
            N_nodes = int(input('Enter the number of nodes: '))

            # create crystal
            crystal = TwoSameImpurities(R, Z_i, Z_p)
            '---Solve equation---'
            (e, de, psi, r, x) = crystal.calculate(N, N_nodes)

            "---Print results and save them to file---"
            '-Print results-'
            print('Energy = ', e)
            print('Energy error = ', de)

            "---Plot results---"
            plt.plot(x, [psi[i][0][0] for i in range(N)], 'r')
            plt.plot(x, [psi[i][1][0] for i in range(N)], 'b')
            # plt.plot(x, [pow(psi[i][0][0], 2) + pow(psi[i][1][0], 2) for i in range(N)], 'b')
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.title(r'$\delta = {0} \lambdabar_c, Z = {1}, R = {2} \lambdabar_c$'.format(crystal.DELTA, Z_i*Z_p, R), \
                      fontsize = 30,  verticalalignment = 'bottom')
            plt.xlabel(r'$x (\lambdabar_c)$', fontsize = 30)
            plt.ylabel(r'$\psi$  ', rotation='horizontal', verticalalignment = 'bottom', fontsize = 30)
            # plot_asymp(e, q, x_max, R)
            plt.show()

            "---Ask user if he wants to save reults---"
            while True:
                choise = input('Save results? (y/n): ')
                if choise == 'y' or choise == 'n':
                    break

            if choise == 'y':
                '-Save results to file-'
                if not os.path.exists('./Results/1D_two_charges'):
                    os.makedirs('./Results/1D_two_charges')

                now = datetime.datetime.now()
                with open('./Results/1D_two_charges/' + crystal.round_seconds(now) + '.txt', 'w') as out:
                    out.write(str(e) + '\n')
                    out.write(str(R/crystal.X_MULT) + '\n')
                    out.write(str(crystal.DELTA/crystal.X_MULT) + '\n')
                    out.write(str(N) + '\n')
                    out.write(str(N_nodes) + '\n\n')
                    for i in range(N):
                        out.write(str(r[i]) + ' ' + str(x[i]) + ' ' + str(psi[i][0][0]) + ' ' + str(psi[i][1][0]) + '\n')

            "---Ask user if he wants to exit program---"
            while True:
                choise = input('Continue? (y/n): ')
                if choise == 'y' or choise == 'n':
                    break

            if choise == 'n':
                break

    @staticmethod
    def asymp_func(x, e, q):
        """Asymptotic wave function for case of two equal charges.

        Args:
            x: point of evaluation
            e: energy of particle
            q = Z_p * Z_i * FINE_STRUCTURE_CONSTANT, where
                Z_p - charge of particle in elementary charge units
                Z_i - charge of impurity in elementary charge units

        Returns:
            Value of asymptotic wave function.

        """
        kor = pow(1 - e*e, 0.5)
        return np.sign(x) * np.exp(-np.abs(x)*kor) * pow(np.abs(x), 2*q*e/kor)

    def plot_asymp(self, e, Z_p, Z_i, x_max, A, Z):
        """Plot asymptotic wave function.

        Args:
            e: energy of particle.
            x_max: right border of plotting. Due to symmetry: x_min = -x_max.

        """
        q = Z_p * Z_i * self.FINE_STRUCTURE_CONSTANT
        r_max = self.x_2_r(x_max)
        r = np.linspace(-r_max, r_max, 1001)
        x = [self.r_2_x(rr) for rr in r]
        C = 0.1 / self.asymp_func(-x_max, e, q)
        f = [C * self.asymp_func(x_i, e, q) for x_i in x]
        plt.plot(x, f, 'g')

    @staticmethod
    def e_from_R(Z_i, Z_p, R_max, N, filename=None):
        """Calculate ground level energy dependence on distance between impurities.

        Args:
            Z_i: impurity charge in elementary charge units.
            Z_p: charge of particle in elementary charge units.
            R_max: maximum impurity position.
            N: number of steps from 0 to R_max.
            filename: path of file to save data to.

        """
        crystal = TwoSameImpurities(0, Z_i, -Z_p)
        e_lst = []
        R_lst = np.linspace(0, R_max, N)
        e_vs_R_data_dir = crystal.delta_dir() + '/e_vs_R/data/'

        for R in R_lst:
            crystal.A = [-R, R]
            (e_i, de_i, psi, r, x) = crystal.calculate(None, 1)
            e_lst.append(e_i)

        '---Save results---'
        # use default filename if not given
        if not filename: filename = e_vs_R_data_dir + str(Z_i) + '.txt'
        with open(filename, 'w') as out:
            out.write(' '.join([str(e) for e in e_lst]) + '\n')
            out.write(' '.join([str(R) for R in R_lst]))

        '---Plot result---'
        crystal.plot_e_from_R(crystal.DELTA, Z_i, R_lst, e_lst)

    @staticmethod
    def plot_e_from_R(dlt, Z_i, R_lst, e_lst):
        """Plot e vs R dependence for impurities with charge Z_i.

        Args:
            dlt: cut-off parameter.
            Z_i: impurity charge.
            R_lst: list of impurity positions.
            e_lst: list of energy values.

        """
        plt.title(r'$\delta = {0} \lambdabar_c, \Z_i = {1}$'.format(dlt, Z_i), \
                  fontsize = 30,  verticalalignment = 'bottom')
        plt.xlabel(r'$R (\lambdabar_c)$', fontsize = 30)
        plt.ylabel(r'$\epsilon$  ', rotation='horizontal', \
                   verticalalignment = 'bottom', horizontalalignment='right', fontsize = 30)
        plt.plot(R_lst, e_lst, 'b')
        plt.show()

    @staticmethod
    def read_e_from_R(path: str):
        """Read data for e vs R plotting from file.

        Args:
            path: path to file with data.

        Returns:
            Tuple of cut-off parameter, impurity charge, list of impurity positions and list of energies.

        """
        dlt = float(re.findall('delta=(.*?)/', path, re.DOTALL)[0])
        Z_i = path[path.rfind('/')+1:]
        Z_i = float(Z_i[:Z_i.rfind('.')])

        with open(path, 'r') as inp:
            R_lst = [float(s) for s in inp.readline().split()]
            e_lst = [float(s) for s in inp.readline().split()]

        return dlt, Z_i, R_lst, e_lst

    @staticmethod
    def e_from_Z(Z_max, R, Z_p, filename=None):
        '''Calculate ground level energy dependence on distance between impurities.

        Args:
            Z_max: maximum value of impurity charge.
            R: impurity position.
            Z_p: charge of particle in elementary charge units.
            filename: path of file to save data to.

        '''
        crystal = TwoSameImpurities(R, 0, Z_p)
        e_lst = []
        Z_lst = [i for i in range(0, Z_max)]
        e_vs_Z_data_dir = crystal.delta_dir() + '/e_vs_Z/data/'

        for Z in Z_lst:
            if Z == 0:
                e_lst.append(1)
            else:
                crystal.Z = [Z, Z]
                (e_i, de_i, psi, r, x) = crystal.calculate(None, 1)
                e_lst.append(e_i)

        '---Save results---'
        # use default filename if not given
        if not filename: filename = e_vs_Z_data_dir + str(R) + '.txt'
        with open(filename, 'w') as out:
            out.write(' '.join([str(e) for e in e_lst]) + '\n')
            out.write(' '.join([str(Z) for Z in Z_lst]))

        '---Plot results---'
        crystal.plot_e_from_Z(crystal.DELTA, R, Z_lst, e_lst)

    @staticmethod
    def plot_e_from_Z(dlt, R, Z_lst, e_lst):
        """Plot e vs R dependence for impurities with charge Z_i.

        Args:
            dlt: cut-off parameter.
            R: impurity position.
            R_lst: list of impurity positions.
            e_lst: list of energy values.

        """
        plt.title(r'$\delta = {0} \lambdabar_c, \R = {1}$'.format(dlt, R), fontsize = 30,  verticalalignment = 'bottom')
        plt.xlabel('Z', fontsize = 30)
        plt.ylabel(r'$\epsilon$  ', rotation='horizontal', \
                   verticalalignment = 'bottom', horizontalalignment='right', fontsize = 30)
        plt.plot(Z_lst, e_lst, 'b')
        plt.show()

    @staticmethod
    def read_e_from_Z(path: str):
        """Read data for e vs Z plotting from file.

        Args:
            path: path to file with data.

        Returns:
            Tuple of cut-off parameter, impurity position, list of impurity charges and list of energies.

        """
        dlt = float(re.findall('delta=(.*?)/', path, re.DOTALL)[0])
        R = path[path.rfind('/')+1:]
        R = float(R[:R.rfind('.')])

        with open(path, 'r') as inp:
            Z_lst = [float(s) for s in inp.readline().split()]
            e_lst = [float(s) for s in inp.readline().split()]

        return dlt, R, Z_lst, e_lst

    def find_Zcr(self, Z_bounds):
        """Find critical charge for given impurity position.

        Args:
            Z_bounds: charge bounds to search in.

        Returns:
            Critical charge.

        """
        while (Z_bounds.up - Z_bounds.down) > 0.01:
            Z = (Z_bounds.up + Z_bounds.down) / 2
            self.Z = [Z, Z]
            (e, de, psi, r, x) = self.calculate(None, 1)
            if e <= -1:
                Z_bounds.up = Z
            else:
                Z_bounds.down = Z

        return Z_bounds.up

    @staticmethod
    def Z_f_R(Z_p, R_lst, Z_lst, nproc, proc_id):
        """Ancillary function for Zcr_from_R.

        Args:
            Z_p: charge of particle in elementary charge units.
            R_lst: list of impurity positions.
            Z_lst: list of critical charge values.
            nproc: number of processes.
            proc_id: current process id.

        """
        N = len(R_lst)
        crystal = TwoSameImpurities(0, 0, Z_p)

        Z_up = Z_lst[-1]

        for i in range(proc_id, N, nproc):
            print(i)
            if i in range(0, nproc):
                Z_down = Z_lst[0]
            else:
                Z_down = Z_lst[i - 1]

            crystal.A = [-R_lst[i], R_lst[i]]
            Z_lst[i] = crystal.find_Zcr(Crystal.Bounds(up = Z_up, down = Z_down))

    @staticmethod
    def Zcr_from_R(Z_p, R_min, R_max, N, filename=None):
        """Calculate critical charge dependence on impurity position, save and plot results.

        Args:
            Z_p: charge of particle in elementary charge units.
            R_min: minimum impurity position.
            R_max: maximum impurity position.
            N: number of points between R_min and R_max.
            filename: path of file to save data to.

        """
        global Zcr_lst
        global R_lst

        nproc = mp.cpu_count()
        processes = []
        manager = mp.Manager()
        Zcr_lst = manager.list([0]*(N+1))
        R_lst = manager.list(np.linspace(R_min, R_max, N + 1))
        Zcr_vs_R_data_dir = TwoSameImpurities.delta_dir() + '/Zcr_vs_R/data/'
        crystal = TwoSameImpurities(0, 0, Z_p)

        crystal.A = [-R_max, R_max]
        Z_up = crystal.find_Zcr(Crystal.Bounds(up = np.pi*0.5 / np.log(0.5/crystal.DELTA), down = 0.001))
        crystal.A = [-R_min, R_min]
        Z_down = crystal.find_Zcr(Crystal.Bounds(up = np.pi*0.5 / np.log(0.5/crystal.DELTA), down = 0.001))

        print("Z_up = ", Z_up)
        print("Z_down = ", Z_down)

        Zcr_lst[0] = Z_down
        Zcr_lst[-1] = Z_up
        for i in range(nproc):
            proc = mp.Process(target=TwoSameImpurities.Z_f_R, args=(Z_p, R_lst, Zcr_lst, nproc, i))
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()

        '---Save results---'
        # use default filename if not given
        if not filename: filename = Zcr_vs_R_data_dir + 'data.txt'
        with open(filename, 'w') as out:
            out.write(' '.join([str(R) for R in R_lst]) + '\n')
            out.write(' '.join([str(Z) for Z in Zcr_lst]))

        '---Plot results---'
        TwoSameImpurities.plot_Zcr_from_R(Crystal.DELTA, R_lst, Zcr_lst)

    @staticmethod
    def read_Zcr_from_R(path: str):
        """Read data for Zcr vs R plotting from file.

        Args:
            path: path to file with data.

        Returns:
            Tuple of cut-off parameter, list of impurity positions and list of impurity critical charges.

        """
        dlt = float(re.findall('delta=(.*?)/', path, re.DOTALL)[0])
        with open(path, 'r') as inp:
            R_lst = [float(s) for s in inp.readline().split()]
            Zcr_lst = [float(s) for s in inp.readline().split()]

        return dlt, R_lst, Zcr_lst

    @staticmethod
    def plot_Zcr_from_R(dlt, R_lst, Zcr_lst):
        """Plot dependence of critical charge from impurity position.

        Args:
            dlt: cut-off parameter.
            R_lst: list of impurity positions.
            Zcr_lst: list of impurity critical charges.

        """
        plt.plot(R_lst, Zcr_lst, 'b')
        plt.title(r'$\delta = {0} \lambdabar_c$'.format(dlt), fontsize = 30,  verticalalignment = 'bottom')
        plt.xlabel(r'$R (\lambdabar_c)$', fontsize = 30)
        plt.ylabel(r'$Z_{cr}$  ', rotation='horizontal', \
                   verticalalignment = 'bottom', horizontalalignment='right', fontsize = 30)
        plt.show()
