#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=continuous
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=312:00:00
#SBATCH --output="out/continuous_train_%A_%a_%j.out"
#SBATCH --error="error/continuous_train_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/lib
export CFLAGS=-I/usr/local/cuda-9.0/include
export LDFLAGS=-L/usr/local/cuda-9.0/lib64
export PATH=$PATH:/usr/local/cuda-9.0/bin
export CUDA_HOME=/usr/local/cuda-9.0
export LIBRARY_PATH=/usr/local/cuda-9.0/lib64

sets=$1
pretrained=$2
bl=$3
emdim=$4
name=$5
typpe=$6
nlayer=$7
tbrd=$8
pre_pos=$9
loss=${10}
pre_more=${11}
src=${12}
orig=${13}

if [ "$pre_pos" == "True" ]
then
  pos=--pos
fi

if [ "$pre_more" == "True" ] 
then
  more=--more
fi

python ${src}/continuous.py --data ${sets}/set_${SLURM_ARRAY_TASK_ID}/\
                            --pretrained ${pretrained}\
                            --save ${bl}_${SLURM_ARRAY_TASK_ID}.pt\
                            --emdim ${emdim}\
                            --log out/${name}_train_${SLURM_ARRAY_TASK_ID}\
                            --name $name\
                            --rnn_type ${typpe}\
                            --nlayers ${nlayer}\
                            --tbrd ${tbrd}\
                            --jobnum ${SLURM_ARRAY_JOB_ID}\
                            --loss ${loss}\
                              ${pos}\
                              ${more}\
                            --orig ${orig}_${SLURM_ARRAY_TASK_ID}.pt
