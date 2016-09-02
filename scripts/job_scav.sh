#!/bin/bash

#SBATCH
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH -t 2:00:00
#SBATCh --nodes=1
#SBATCH --mem=8g

cd ~/Code/moving_poselet
mkdir -p ~/logfiles/mp_journal/$1/$4
mkdir -p ~/scratch/mp_journal/$1/$4

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,base_compiledir=/home-3/ltao4@jhu.edu/scratch/${SLURM_JOB_ID} python moving_poselet_exp_set.py $1 $2 $3 --exp $4 -s $5 --l2 $6 --l1 $7 --rs $8 --epoch 150 > ~/logfiles/mp_journal/$1/$4/${SLURM_JOB_ID}_nword$2_layer$3_multi$4_l1$5_l2$6_rs$7.out
