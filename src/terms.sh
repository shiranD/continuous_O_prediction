#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output="out/terms_%A_%a_%j.out"
#SBATCH --error="error/terms_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

model=$1
set1=$2
output=$3
name=$4
src=$5

mkdir -p ${output}
# how many files to process
ls ${model}/preds | wc -l > ln_num
number=$(cat ln_num)
number=$(($number/3+1))
echo $number

python ${src}/terms.py --data ${set1}\
                       --outdir ${output}\
                       --name ${name}\
                       --num ${number}\
                       --inference_folder ${model}
