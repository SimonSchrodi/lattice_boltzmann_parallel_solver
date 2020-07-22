# Lattice Boltzmann Parallel Solver

## Results

### Von Karman vortex street
<img src="https://raw.githubusercontent.com/infomon/lattice_boltzmann_parallel_solver/master/figures/von_karman_vortex_shedding/png_to_gif.gif" />

### Couette flow evolution
![Couette flow](figures/couette_flow/vel_vectors_evolution.svg)

### Poiseuille flow evolution
![Poiseuille flow](figures/poiseuille_flow/vel_vectors_evolution.svg)

## High-level structure
The code is organized as follows:
- ![documentation](documentation) contains the report
- ![figures](figures) contains figures from our experiments
- ![src](src) contains the main part of the code
  - ![src/boundary_conditions.py](src/boundary_conditions.py) implements several boundary conditions
  - ![src/boundary_utils.py](src/boundary_utils) plugs specific combinations of boundary conditions together
  - ![src/experiments.py](src/experiments.py) contains functions to run the experiments
  - ![src/initial_values.py](src/initial_values.py) contains initial value specifications
  - ![src/lattice_boltzmann_method.py](src/lattice_boltzmann_method.py) implements the basic ingredients of the LBM
  - ![src/main.py](src/main.py) calls the experiments specified by the command arguments
  - ![src/parallelization_utils.py](src/parallelization_utils.py) implements essential parts of the parallel communication
  - ![src/visualizations_utils.py](src/visualizations_utils.py) contains several helper functions for visualizations
- ![tests](tests) contains several unit tests for Travis CI

## How to build
`make init`

## How to reproduce results
- Shear wave decay
  - `python src/main.py -f "shear_wave_decay_density"`
  - `python src/main.py -f "shear_wave_decay_velocity"`
  - `python src/main.py -f "viscosity_vs_omega"`
- Couette flow
  - `python src/main.py -f "couette_vectors"`
  - `python src/main.py -f "couette_evolution"`
- Poiseuille flow
  - `python src/main.py -f "poiseuille_vectors"`
  - `python src/main.py -f "poiseuille_evolution"`
- von Karman vortex street
  - `python src/main.py -f "reynold_strouhal"`
  - `python src/main.py -f "nx_strouhal"`
  - `python src/main.py -f "blockage_strouhal"`
- scaling tests
  - `mpirun -N 4 python src/main.py -f "scaling_test"`
- pngs images to gif
  - `python src/main.py -f "pngs_to_gif"`
