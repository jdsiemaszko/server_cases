#!/bin/bash
#
#SBATCH --job-name="Re=100,a=0.5"
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-ae-msc-ae

module load 2022r2 
module load cuda/11.7
module load openmpi/4.1.1

python run.py config/config_vrm.yaml > output_vrm.log
python run.py config/config_avrm.yaml > output_avrm.log
python run.py config/config_tutty.yaml > output_tutty.log
python run.py config/config_merge_avrm.yaml > output_merge_avrm.log
python run.py config/config_remesh_avrm.yaml > output_remesh_avrm.log

