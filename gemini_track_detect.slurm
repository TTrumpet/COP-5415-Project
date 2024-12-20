#!/bin/bash
#SBATCH --account=mshah
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH -c 20
#SBATCH --mem-per-cpu=24G
#SBATCH --time=48:00:00
#SBATCH --job-name=perception_gemini
#SBATCH -e slurm_output-%j.err
#SBATCH -o slurm_output-%j.out


echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo $SLURM_JOB_NODELIST

module load anaconda/anaconda-2023.09
source activate sf_llava
module load cuda/cuda-12.1.0
module load gcc/gcc-12.2.0

export SAM2_REPO_ROOT=./sam2
export PYTHONPATH="${SAM2_REPO_ROOT}:${PYTHONPATH}"

### Running the program ###
nvidia-smi && time srun python gemini_vqa_tracking_owl_singleobj.py

