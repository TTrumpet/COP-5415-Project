#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH -c 20
#SBATCH --mem-per-cpu=24G
#SBATCH --time=72:00:00
#SBATCH --job-name=perception_lsf
#SBATCH -e gemini_output-%j.err
#SBATCH -o gemini_output-%j.out


echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo $SLURM_JOB_NODELIST

module load anaconda/anaconda-2023.09
source activate sf_llava
module load cuda/cuda-12.1.0
module load gcc/gcc-12.2.0

## export PYTHONPATH="./:$PYTHONPATH"

### Running the program ###
nvidia-smi && time srun python run_perception.py

