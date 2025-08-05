#!/bin/bash
#SBATCH --job-name=score-roberta-prior
#SBATCH --output /n/netscratch/sham_lab/Everyone/cbrownpinilla/hyperfilter/hyperbolic-transformer/logs/pretrain/output/%x-%A_%a.log
#SBATCH -p kempner_h100
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=950GB		
#SBATCH --array=1-1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=camilobrownpinilla@college.harvard.edu

module load python/3.10.13-fasrc01
conda deactivate
conda activate hyperfilter


export CHECKPOINTS_PATH=/n/netscratch/sham_lab/Everyone/cbrownpinilla/scores/roberta/prior

export CONFIG=configs/base.yaml

# Accept sweep config as argument
export OVERWRITE_CONFIG=$1

srun python score.py configs/base.yaml --overwrites ${OVERWRITE_CONFIG}