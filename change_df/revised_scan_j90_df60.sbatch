#!/bin/bash -e

#SBATCH --mail-user=jyar540@aucklanduni.ac.nz
#SBATCH --mail-type=ALL
#SBATCH --job-name=EnergyScanJ90DF60
#SBATCH --account=uoa00623
#SBATCH --output=simple.out
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1

module load Julia/1.5.1-GCC-9.2.0-VTune

julia revised_j90_df60.jl
