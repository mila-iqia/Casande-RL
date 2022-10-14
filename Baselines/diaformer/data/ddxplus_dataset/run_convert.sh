#!/bin/bash
#SBATCH --job-name=convert_casande_data
#SBATCH --output=/home/mila/z/zhi.wen/slurm_outputs/output_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20Gb

export PYTHONUNBUFFERED=1

module load miniconda/3

conda activate med_evi_diaformer

python convert_to_diaformer_format.py \
    --train_data_path "/network/data2/amlrt_internships/automatic-medical-evidence-collection/data/release_train_patients.zip" \
    --val_data_path "/network/data2/amlrt_internships/automatic-medical-evidence-collection/data/release_validate_patients.zip" \
    --test_data_path "/network/data2/amlrt_internships/automatic-medical-evidence-collection/data/release_test_patients.zip" \
    --evi_meta_path "/home/mila/z/zhi.wen/medical_evidence_collection/automated-medical-evidence-collection/data/evidences.json" \
    --save_dir "/home/mila/z/zhi.wen/medical_evidence_collection/diaformer_data/casande_type_dataset" \