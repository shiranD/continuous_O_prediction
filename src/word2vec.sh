#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=embedding_generator
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/generate_"%A_%a_%j.out"
#SBATCH --error=error/generate_"%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"


filename=$1
modelname=$2
emdim=$3
traindir=$4

python ${traindir}/w2v_generator.py --filename ${filename} --modelname ${modelname} --emdim ${emdim}
