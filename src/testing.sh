#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output="out/test_%A_%a_%j.out"
#SBATCH --error="error/test_%A_%a_%j.err"
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
gen=$3
emdim=$4
kwd=$5
name=$6
pre_pos=$7
core=$8
src=$9

if [ "$pre_pos" == "True" ]
then
  pos=--pos
fi

SLURM_ARRAY_TASK_ID=0

python ${src}/testing.py --name ${name}\
                         --data ${sets}/set_${SLURM_ARRAY_TASK_ID}/\
                         --pretrained ${pretrained}\
                         --save ${gen}_${SLURM_ARRAY_TASK_ID}.pt\
                         --emdim ${emdim}\
                         --core ${core}\
                         ${pos}
