from experiments import \
    (plot_evolution_of_density,
     plot_evolution_of_velocity,
     plot_measured_viscosity_vs_omega,
     plot_couette_flow_evolution,
     plot_couette_flow_vel_vectors,
     plot_poiseuille_flow_vel_vectors,
     plot_poiseuille_flow_evolution)

from milestoneQuickFunctionCalls import milestone_6

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--function", type=str, choices=["shear_wave_decay_velocity",
                                                               "shear_wave_decay_density",
                                                               "viscosity_vs_omega",
                                                               "couette_evolution",
                                                               "couette_vectors",
                                                               "poiseuille_vectors",
                                                               "poiseuille_evolution",
                                                               "Reynold_Strouhal"
                                                               "scaling_test"],
                        help="Which figure to generate")
    args = parser.parse_args()

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
    elif args.function == "Reynold_Strouhal":
        None
    elif args.function == "scaling_test":
        None
    else:
        raise Exception('Unknown function')


if __name__ == "__main__":
    main()
