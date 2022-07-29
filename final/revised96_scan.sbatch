#!/bin/bash -e

#SBATCH --mail-user=jyar540@aucklanduni.ac.nz
#SBATCH --mail-type=ALL
#SBATCH --job-name=EnergyScan96
#SBATCH --account=uoa00623
#SBATCH --output=simple.out
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-169
#SBATCH --cpus-per-task=1

module load Julia/1.5.1-GCC-9.2.0-VTune

julia revised96.jl
