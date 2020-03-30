#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=300:00:00
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
name=$5
pre_pos=$6
core=$7
src=$8

if [ "$pre_pos" == "True" ]
then
  pos=--pos
fi

mkdir -p ${name}
mkdir -p ${name}/preds
mkdir -p ${name}/pos
mkdir -p ${name}/results

if [${core} == traditional ]
then
  python ${src}/bl_predictions.py --name ${name}\
                               --data ${sets}/set_${SLURM_ARRAY_TASK_ID}/\
                               --pretrained ${pretrained}\
                               --save ${gen}_${SLURM_ARRAY_TASK_ID}.pt\
                               --emdim ${emdim}\
                               ${pos}
else
  python ${src}/predictions.py --name ${name}\
                               --data ${sets}/set_${SLURM_ARRAY_TASK_ID}/\
                               --pretrained ${pretrained}\
                               --save ${gen}_${SLURM_ARRAY_TASK_ID}.pt\
                               --emdim ${emdim}\
                               ${pos}
  python ${src}/tk_predictions.py --name ${name}\
                               --data ${sets}/set_${SLURM_ARRAY_TASK_ID}/\
                               --pretrained ${pretrained}\
                               --save ${gen}_${SLURM_ARRAY_TASK_ID}.pt\
                               --emdim ${emdim}\
                               ${pos}
fi
