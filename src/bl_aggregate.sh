#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output="out/bl_aggregate_%A_%a_%j.out"
#SBATCH --error="error/bl_aggregate_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

name=$1
outdir=$2
kwd=$3
src=$4

# figure out how many files (fix divide by 3)
ls $name/preds | wc -l > ln_num
number=$(cat ln_num)
echo $number

python ${src}/bl_aggregate.py --folder ${name} --number ${number} --sett ${kwd} --termdir ${outdir} 
