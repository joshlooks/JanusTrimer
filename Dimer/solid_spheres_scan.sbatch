#!/bin/bash -e

#SBATCH --mail-user=ctod705@aucklanduni.ac.nz
#SBATCH --mail-type=ALL
#SBATCH --job-name=SolidScan
#SBATCH --account=uoa00623
#SBATCH --output=simple.out
#SBATCH --ntasks=1-10
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --array=1-37

module load Julia/1.5.1-GCC-9.2.0-VTune

julia solidscan.jl