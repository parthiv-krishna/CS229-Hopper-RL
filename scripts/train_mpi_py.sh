#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name mpi_trainer
#SBATCH -o mpi/mpi_%j.log
#SBATCH -N 1
#SBATCH -n 17
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:2
#SBATCH --time=2-00:00:00

# Initialize the module command first
source /etc/profile

# Load MPI module
module load anaconda/2020a
module load mpi/openmpi-4.0

# Call your script as you would from the command line
echo "mpirun python -B python/train_agent.py CarRacing-cubic-v1 mppi --nsteps 500000"
mpirun python -B python/train_agent.py CarRacing-cubic-v1 mppi --nsteps 500000

