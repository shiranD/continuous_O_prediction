#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output="out/aggregate_%A_%a_%j.out"
#SBATCH --error="error/aggregate_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

name=$1
dir=$2
pre_pos=$3
kwd=$4
src=$5

if [ "$pre_pos" == "True" ]
then
  pos=--pos
fi

# figure out how many files
ls $name/preds | wc -l > ln_num
number=$(($number/3+1))

python ${src}/aggregate.py --folder ${name} --number ${number} --termdir ${dir} ${pos} --sett ${kwd}
