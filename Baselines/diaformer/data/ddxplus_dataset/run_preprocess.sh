#!/bin/bash
#SBATCH --job-name=preprocess_casande_data
#SBATCH --output=/home/mila/z/zhi.wen/slurm_outputs/output_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20Gb

export PYTHONUNBUFFERED=1

module load miniconda/3

conda activate med_evi_diaformer

python preprocess.py