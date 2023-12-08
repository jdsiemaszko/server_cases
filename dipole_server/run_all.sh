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

python run.py config/config_VRM.yaml > output_VRM.log
python run.py config/config_AVRM.yaml > output_AVRM.log
python run.py config/config_merge_AVRM.yaml > output_merge_AVRM.log
python run.py config/config_remesh_AVRM.yaml > output_remesh_AVRM.log