#!/bin/bash

#SBATCH
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCh --nodes=1
#SBATCH --mem=8g

cd ~/Code/moving_poselet
mkdir -p ~/logfiles/mp_journal/$1/$4/
mkdir -p ~/scratch/mp_journal/$1/$4/

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,base_compiledir=/home-3/ltao4@jhu.edu/scratch/${SLURM_JOB_ID} python moving_poselet_exp.py $1 $2 $3 --epoch 200 --full --split $4 --sub $5 --f Ges_Feature