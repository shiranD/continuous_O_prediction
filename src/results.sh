#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output="out/results_%A_%a_%j.out"
#SBATCH --error="error/results_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

sets=$1
pretrained=$2
gen=$3
emdim=$4
name=$5
pre_pos=$6
src=$7

if [ "$pre_pos" == "True" ]
then
  pos=--pos
fi

python ${src}/results.py --name ${name}\
                         --data ${sets}/set_0/\
                         --pretrained ${pretrained}\
                         ${pos}\
                         --emdim ${emdim}\
                         --number ${SLURM_ARRAY_TASK_ID} 
