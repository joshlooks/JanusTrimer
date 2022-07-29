#!/bin/bash -e

#SBATCH --mail-user=ctod705@aucklanduni.ac.nz
#SBATCH --mail-type=ALL
#SBATCH --job-name=EnergyScan
#SBATCH --account=uoa00623
#SBATCH --output=simple.out
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-702

module load Python

python fast_ljp_scan.py
