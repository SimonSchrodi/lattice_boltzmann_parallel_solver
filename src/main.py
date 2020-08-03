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

import inspect


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


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
                        required=True,
                        help="Which figure to generate")
    parser.add_argument("-l", "--lattice_grid_size", type=int, nargs='+', help="Lattice grid size")
    parser.add_argument("-o", "--omega", type=float, help="Relaxation parameter")
    parser.add_argument("-t", "--time_steps", type=int, help="Time steps for simulation")
    parser.add_argument("-ep", "--epsilon_p", type=float, help="Amplitude of sinusoidal initial condition for density")
    parser.add_argument("-ev", "--epsilon_v", type=float, help="Amplitude of sinusoidal initial condition for velocity")
    parser.add_argument("-p0", "--initial_p0", type=float, help="Average density of sinusoidal initial condition")
    parser.add_argument("-viz", "--nof_viz", type=int,
                        help="Number of visualizations to plot. Has to be divisible by 5")
    parser.add_argument("-od", "--omega_discretization", type=int, help="Discretization of omega")
    parser.add_argument("-mwv", "--moving_wall_vel", type=float, help="Velocity of moving wall")
    parser.add_argument("-dp", "--delta_p", type=float, help="Pressure difference inlet and outlet")
    parser.add_argument("-ps", "--plate_size", type=int, help="Size of the plate")
    parser.add_argument("-id", "--inlet_den", type=float, help="Inlet velocity")
    parser.add_argument("-iv", "--inlet_vel", type=float, help="Inlet density")
    parser.add_argument("-nu", "--kinematic_visc", type=float, help="Kinematic viscosity")
    args = parser.parse_args()

    if args.lattice_grid_size is not None:
        lattice_grid_size = tuple(args.lattice_grid_size)

    if args.function == "shear_wave_decay_density":
        default_params = list(get_default_args(plot_evolution_of_density).values())
        plot_evolution_of_density(
            lattice_grid_shape=lattice_grid_size if args.lattice_grid_size is not None else default_params[0],
            initial_p0=args.initial_p0 if args.initial_p0 is not None else default_params[1],
            epsilon=args.epsilon_p if args.epsilon_p is not None else default_params[2],
            omega=args.omega if args.omega is not None else default_params[3],
            time_steps=args.time_steps if args.time_steps is not None else default_params[4],
            number_of_visualizations=args.nof_viz if args.nof_viz is not None else default_params[5]
        )
    elif args.function == "shear_wave_decay_velocity":
        default_params = list(get_default_args(plot_evolution_of_velocity).values())
        plot_evolution_of_velocity(
            lattice_grid_shape=lattice_grid_size if args.lattice_grid_size is not None else default_params[0],
            epsilon=args.epsilon_v if args.epsilon_v is not None else default_params[1],
            omega=args.omega if args.omega is not None else default_params[2],
            time_steps=args.time_steps if args.time_steps is not None else default_params[3],
            number_of_visualizations=args.nof_viz if args.nof_viz is not None else default_params[4]
        )
    elif args.function == "viscosity_vs_omega":
        default_params = list(get_default_args(plot_measured_viscosity_vs_omega).values())
        plot_measured_viscosity_vs_omega(
            lattice_grid_shape=lattice_grid_size if args.lattice_grid_size is not None else default_params[0],
            initial_p0=args.initial_p0 if args.initial_p0 is not None else default_params[1],
            epsilon_p=args.epsilon_p if args.epsilon_p is not None else default_params[2],
            epsilon_v=args.epsilon_v if args.epsilon_v is not None else default_params[3],
            time_steps=args.time_steps if args.time_steps is not None else default_params[4],
            omega_discretization=args.omega_discretization if args.omega_discretization is not None else default_params[
                5],
        )
    elif args.function == "couette_evolution":
        default_params = list(get_default_args(plot_couette_flow_evolution).values())
        plot_couette_flow_evolution(
            lattice_grid_shape=lattice_grid_size if args.lattice_grid_size is not None else default_params[0],
            omega=args.omega if args.omega is not None else default_params[1],
            U=args.moving_wall_vel if args.moving_wall_vel is not None else default_params[2],
            time_steps=args.time_steps if args.time_steps is not None else default_params[3],
            number_of_visualizations=args.nof_viz if args.nof_viz is not None else default_params[4]
        )
    elif args.function == "couette_vectors":
        default_params = list(get_default_args(plot_couette_flow_vel_vectors).values())
        plot_couette_flow_vel_vectors(
            lattice_grid_shape=lattice_grid_size if args.lattice_grid_size is not None else default_params[0],
            omega=args.omega if args.omega is not None else default_params[1],
            U=args.moving_wall_vel if args.moving_wall_vel is not None else default_params[2],
            time_steps=args.time_steps if args.time_steps is not None else default_params[3]
        )
    elif args.function == "poiseuille_evolution":
        default_params = list(get_default_args(plot_poiseuille_flow_evolution).values())
        plot_poiseuille_flow_evolution(
            lattice_grid_shape=lattice_grid_size if args.lattice_grid_size is not None else default_params[0],
            omega=args.omega if args.omega is not None else default_params[1],
            delta_p=args.delta_p if args.delta_p is not None else default_params[2],
            time_steps=args.time_steps if args.time_steps is not None else default_params[3],
            number_of_visualizations=args.nof_viz if args.nof_viz is not None else default_params[4]
        )
    elif args.function == "poiseuille_vectors":
        default_params = list(get_default_args(plot_poiseuille_flow_vel_vectors).values())
        plot_poiseuille_flow_vel_vectors(
            lattice_grid_shape=lattice_grid_size if args.lattice_grid_size is not None else default_params[0],
            omega=args.omega if args.omega is not None else default_params[1],
            delta_p=args.delta_p if args.delta_p is not None else default_params[2],
            time_steps=args.time_steps if args.time_steps is not None else default_params[3],
        )
    elif args.function == "plot_von_karman":
        default_params = list(get_default_args(plot_parallel_von_karman_vortex_street).values())
        plot_parallel_von_karman_vortex_street(
            lattice_grid_shape=lattice_grid_size if args.lattice_grid_size is not None else default_params[0],
            plate_size=args.plate_size if args.plate_size is not None else default_params[1],
            inlet_density=args.inlet_den if args.inlet_den is not None else default_params[2],
            inlet_velocity=args.inlet_vel if args.inlet_vel is not None else default_params[3],
            kinematic_viscosity=args.kinematic_visc if args.kinematic_visc is not None else default_params[4],
            time_steps=args.time_steps if args.time_steps is not None else default_params[5]
        )
    elif args.function == "reynold_strouhal":
        default_params = list(get_default_args(x_strouhal).values())
        reynolds_numbers = [40, 70, 100, 140, 170, 200]
        for re in reynolds_numbers:
            if re > 130:
                visc = 0.03
                vel = re * visc / 40
                x_strouhal(
                    folder_name='reynold_strouhal',
                    lattice_grid_shape=lattice_grid_size if args.lattice_grid_size is not None else default_params[0],
                    plate_size=args.plate_size if args.plate_size is not None else default_params[1],
                    inlet_density=args.inlet_den if args.inlet_den is not None else default_params[2],
                    inlet_velocity=vel,
                    kinematic_viscosity=visc,
                    time_steps=args.time_steps if args.time_steps is not None else default_params[5]
                )
            else:
                visc = 40 * 0.1 / re
                x_strouhal(
                    folder_name='reynold_strouhal',
                    lattice_grid_shape=lattice_grid_size if args.lattice_grid_size is not None else default_params[0],
                    plate_size=args.plate_size if args.plate_size is not None else default_params[1],
                    inlet_density=args.inlet_den if args.inlet_den is not None else default_params[2],
                    inlet_velocity=default_params[3],
                    kinematic_viscosity=visc,
                    time_steps=args.time_steps if args.time_steps is not None else default_params[5]
                )
        plot_reynolds_strouhal()
    elif args.function == "nx_strouhal":
        default_params = list(get_default_args(x_strouhal).values())
        lxs = [260, 300, 350, 420, 500, 750, 1000, 1250]
        ly = 180
        for lx in lxs:
            x_strouhal(
                folder_name='nx_strouhal',
                lattice_grid_shape=(lx, ly),
                plate_size=args.plate_size if args.plate_size is not None else default_params[1],
                inlet_density=args.inlet_den if args.inlet_den is not None else default_params[2],
                inlet_velocity=args.inlet_vel if args.inlet_vel is not None else default_params[3],
                kinematic_viscosity=args.kinematic_visc if args.kinematic_visc is not None else default_params[4],
                time_steps=args.time_steps if args.time_steps is not None else default_params[5]
            )
        plot_nx_strouhal()
    elif args.function == "blockage_strouhal":
        default_params = list(get_default_args(x_strouhal).values())
        lx = 420
        lys = [60, 100, 140, 180, 260]
        for ly in lys:
            x_strouhal(
                folder_name='blockage_strouhal',
                lattice_grid_shape=(lx, ly),
                plate_size=args.plate_size if args.plate_size is not None else default_params[1],
                inlet_density=args.inlet_den if args.inlet_den is not None else default_params[2],
                inlet_velocity=args.inlet_vel if args.inlet_vel is not None else default_params[3],
                kinematic_viscosity=args.kinematic_visc if args.kinematic_visc is not None else default_params[4],
                time_steps=args.time_steps if args.time_steps is not None else default_params[5]
            )
        plot_blockage_strouhal()
    elif args.function == "scaling_test":
        default_params = list(get_default_args(scaling_test).values())
        scaling_test(
            folder_name='scaling_test',
            lattice_grid_shape=lattice_grid_size if args.lattice_grid_size is not None else default_params[0],
            plate_size=args.plate_size if args.plate_size is not None else default_params[1],
            inlet_density=args.inlet_den if args.inlet_den is not None else default_params[2],
            inlet_velocity=args.inlet_vel if args.inlet_vel is not None else default_params[3],
            kinematic_viscosity=args.kinematic_visc if args.kinematic_visc is not None else default_params[4],
            time_steps=args.time_steps if args.time_steps is not None else default_params[5]
        )
        plot_scaling_test()
    elif args.function == "pngs_to_gif":
        pngs_to_gif()
    else:
        raise Exception('Unknown function')


if __name__ == "__main__":
    main()
