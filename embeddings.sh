#!/bin/bash
set -x
set -e

# this is an example for generating word embeddigns with word2vec

main=TBD # path to Shiran's main dir /home/groups/BedrickLab/shiran/gans/ 

# Data
sett=${main}/data # path to main data folder
kwd=pos_pb_big # subdir in main data folder 
sets=${sett}/${kwd}/set_0 # the full datapath
emdim=50
modelname=embeddings/w2v_${emdim}_${kwd}

# SRC code
traindir=src #path to source code for training

mkdir -p embeddings

# concat all files
cat ${sets}/train ${sets}/test ${sets}/valid > ${sets}/full
filename=${sets}/full

# comment cat and uncomment sbatch after cat was run
#sbatch --job-name word2vec ${traindir}/word2vec.sh ${filename} ${modelname} ${emdim} ${traindir}
