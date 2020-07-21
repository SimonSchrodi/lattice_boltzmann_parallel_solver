#!/bin/bash -x
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:40:00
#SBATCH -J HPC_WITH_PYTHON_Reynold_Strouhal
#SBATCH --mem=6gb
#SBATCH --export=ALL
#SBATCH --partition=multiple

module load devel/python/3.8.1_gnu_9.2-pipenv
module load mpi/openmpi/4.0

echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_JOB_CPUS_PER_NODE} cores each."
echo "Each node has ${SLURM_MEM_PER_NODE} of memory allocated to this job."
time mpirun python src/main.py -f "reynold_strouhal"
