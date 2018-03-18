from two_identical_impurities import TwoSameImpurities


if __name__ == '__main__':
    crystal = TwoSameImpurities(0.1, 10, -1)
    # crystal.create_file_tree()
    # crystal.data_for_LDOS(12)
    crystal.plot_LDOS(list(range(2, 13)))


# TwoSameImpurities.user_interface()
# crystal = TwoSameImpurities(0.1, 10, -1)
# crystal.create_file_tree()
# crystal.data_for_LDOS(12)
# crystal.plot_LDOS()

# l = [(float(s.split()[0]), float(s.split()[1])) for s in open('18.txt', 'r').readlines()[1:]]
# x, y = zip(*l)
# plt.plot(x, y, 'b')
# plt.show()

# R_lst = np.linspace(0, 1, 11)
# args = [(14, 1, 7, r) for r in R_lst]
#
# p = mp.Pool(3)
# p.starmap(data_for_LDOS, args)
# A = [-0.1, 0.1]
# Z = [10, 10]
# create_file_tree(A, Z)
# Zcr_from_R(-1, 0, 5, 40)
# data_for_LDOS(12, -1, A, Z)

# energies = find_energy(10, Bounds(up=1.0-1e-10, down=-1.0+1e-10), -1, A, Z, None)
# print(energies)
# for e in energies:
#     psi, x, _ = calculate_for_given_energy(e, -1, A, Z, 1)
#     plot_wave_function(psi, x)

# plot_level('./Results/alpha=0.00729735256672/delta=0.01/(-0.1,10) (0.1,10)/levels/data/8.txt')
# plot_for_given_energy()
# find_energy()
# e_from_R()
# e_from_Z()
# Zcr_from_R(0, 15, 300)
# plot_Zcr_from_R('./Results/Zcr_from_R/18.txt')
# print(find_Zcr(0, Bounds(up=20, down=0.0001)))
