# Lattice Boltzmann Parallel Solver

## Results

### Von Karman vortex street
<img src="https://raw.githubusercontent.com/infomon/lattice_boltzmann_parallel_solver/master/figures/von_karman_vortex_shedding/png_to_gif.gif" />

### Couette flow evolution
![Couette flow](figures/couette_flow/vel_vectors_evolution.svg)

### Poiseuille flow evolution
![Poiseuille flow](figures/poiseuille_flow/vel_vectors_evolution.svg)

## How to reproduce results
- Shear wave decay
  - `python src/main.py -f shear_wave_decay_density`
  - `python src/main.py -f shear_wave_decay_velocity`
  - `python src/main.py -f viscosity_vs_omega`
- Couette flow
  - `python src/main.py -f couette_vectors`
  - `python src/main.py -f couette_evolution`
- Poiseuille flow
  - `python src/main.py -f poiseuille_vectors`
  - `python src/main.py -f poiseuille_evolution`
- von Karman vortex street
  - `python src/main.py -f reynold_strouhal`
- scaling tests
  - `python src/main.py -f scaling_test`
