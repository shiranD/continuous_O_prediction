#!/bin/bash
set -x
set -e

# SRC code
traindir=src #path to source code for training
sett=pos_pb_big
set1=../../../data/${sett}/set_0/
outdir=sets_${sett}
ZERO=$(sbatch --job-name=terms $traindir/terms.sh ${set1} ${outdir} ${sett} ${traindir} | cut -f 4 -d' ')
