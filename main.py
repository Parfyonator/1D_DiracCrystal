from two_identical_impurities import TwoSameImpurities


if __name__ == '__main__':
    # create crystal with two same impurities
    crystal = TwoSameImpurities(0.1, 10, -1)
    # create file tree
    crystal.create_file_tree()
    # calculate data for LDOS
    crystal.data_for_LDOS(12)
    # plot LDOS for levels from 2 to 12
    crystal.plot_LDOS(list(range(2, 13)))

    # calculate and plot dependence of lowest energy level from impurity position
    TwoSameImpurities.e_from_R(10, -1, 0, 10, 10)
    # calculate and plot dependence of lowest energy level from impurity charge
    TwoSameImpurities.e_from_Z(1, 15, 0.1, -1, 20)
    # calculate and plot dependence of critical charge from impurity position
    TwoSameImpurities.Zcr_from_R(-1, 0, 10, 100)
