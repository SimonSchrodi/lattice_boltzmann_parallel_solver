from experiments import (
    plot_evolution_of_density,
    plot_evolution_of_velocity,
    plot_measured_viscosity_vs_omega,
    plot_couette_flow_evolution,
    plot_couette_flow_vel_vectors,
    plot_poiseuille_flow_vel_vectors,
    plot_poiseuille_flow_evolution,
    plot_parallel_von_karman_vortex_street,
    x_strouhal,
    scaling_test
)

from visualizations_utils import (
    pngs_to_gif,
    plot_reynolds_strouhal,
    plot_nx_strouhal,
    plot_blockage_strouhal,
    plot_scaling_test
)

import argparse


def main():
    """
    Main function directing command line arguments to experiments/plotting utilities
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--function", type=str, choices=["shear_wave_decay_velocity",
                                                               "shear_wave_decay_density",
                                                               "viscosity_vs_omega",
                                                               "couette_evolution",
                                                               "couette_vectors",
                                                               "poiseuille_vectors",
                                                               "poiseuille_evolution",
                                                               "plot_von_karman",
                                                               "reynold_strouhal",
                                                               "nx_strouhal",
                                                               "blockage_strouhal",
                                                               "scaling_test",
                                                               "pngs_to_gif"],
                        help="Which figure to generate")
    parser.add_argument("-lx", "--lx", type=int)
    parser.add_argument("-ly", "--ly", type=int)
    args = parser.parse_args()

    if args.lx is not None or args.ly is not None:
        raise Exception('lx and ly both have to be defined')

    if args.lx is not None and args.ly is not None:
        lx = args.lx
        ly = args.ly

    if args.function == "shear_wave_decay_density":
        plot_evolution_of_density()
    elif args.function == "shear_wave_decay_velocity":
        plot_evolution_of_velocity()
    elif args.function == "viscosity_vs_omega":
        plot_measured_viscosity_vs_omega()
    elif args.function == "couette_evolution":
        plot_couette_flow_evolution()
    elif args.function == "couette_vectors":
        plot_couette_flow_vel_vectors()
    elif args.function == "poiseuille_evolution":
        plot_poiseuille_flow_evolution()
    elif args.function == "poiseuille_vectors":
        plot_poiseuille_flow_vel_vectors()
    elif args.function == "plot_von_karman":
        if args.lx is not None and args.ly is not None:
            plot_parallel_von_karman_vortex_street(lattice_grid_shape=(lx, ly))
        else:
            plot_parallel_von_karman_vortex_street()
    elif args.function == "reynold_strouhal":
        reynolds_numbers = [40, 70, 100, 140, 170, 200]
        for re in reynolds_numbers:
            if re > 130:
                visc = 0.03
                vel = re * visc / 40
                x_strouhal(folder_name='reynold_strouhal', inlet_velocity=vel, kinematic_viscosity=visc,
                           time_steps=200000)
            else:
                visc = 40*0.1/re
                x_strouhal(folder_name='reynold_strouhal', kinematic_viscosity=visc, time_steps=200000)
        plot_reynolds_strouhal()
    elif args.function == "nx_strouhal":
        lxs = [260, 300, 350, 420, 500, 750, 1000, 1250]
        ly = 180
        for lx in lxs:
            x_strouhal(folder_name='nx_strouhal', lattice_grid_shape=(lx, ly), time_steps=200000)
        plot_nx_strouhal()
    elif args.function == "blockage_strouhal":
        lx = 420
        lys = [60, 100, 140, 180, 260]
        for ly in lys:
            x_strouhal(folder_name='blockage_strouhal', lattice_grid_shape=(lx, ly))
        plot_blockage_strouhal()
    elif args.function == "scaling_test":
        if args.lx is not None and args.ly is not None:
            scaling_test(folder_name='scaling_test', lattice_grid_shape=(lx, ly))
            plot_scaling_test((lx, ly))
        else:
            scaling_test(folder_name='scaling_test')
            plot_scaling_test()
    elif args.function == "pngs_to_gif":
        pngs_to_gif()
    else:
        raise Exception('Unknown function')


if __name__ == "__main__":
    main()
