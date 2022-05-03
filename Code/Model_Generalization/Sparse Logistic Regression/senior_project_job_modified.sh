#!/bin/bash
#SBATCH -J senior_thesis_program
#SBATCH --output=print_output_modified.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 2-00:00:00

module load CUDA
module load cuDNN
source env/bin/activate
python sim_num_samples_modified.py
