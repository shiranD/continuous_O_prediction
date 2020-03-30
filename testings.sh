#!/bin/bash
set -x
set -e

main=../../.. # path to Shiran's main dir /home/groups/BedrickLab/shiran/gans/

# Data
sett=${main}/data # path to main data folder
emkwd=pos_nyt_big # embd used
kwd=pos_nyt_big # (pos_nyt, pos_nyt_big) set used
sets=${sett}/${kwd} # the full datapath

# SRC code
traindir=src #path to source code for training

# Embedding
embeddingpath=${main}/embeddings # path to embedding dir
#T=_T # (values: _t or nothing) if trans happens add it to the name
emset=w2v # type/name of embedding file
emdim=50 # embedding dimension

# Model architecture
typpe=LSTM #core RNN layer
nlayer=2 # number of layers
loss=cos # type of loss function (cos, mse, mm). irrelevant for traditional training (variable is ignored)
pos=True # (False/True values) apply POS decoding, irrelevant for traditional training (variable is ignored)

# Model Core approach
core=continuous
#core=gan
#core=traditional

# Model name and path
kwd=pos_nyt_big # model used
modelpath=${kwd}_models # dirname of models (will be created)
name=${loss}_${typpe}${nlayer}${T}_${core}_${emdim}_${kwd}_${emset} # training name
save=$modelpath/${name} # modelname
setdir=sets_${kwd}/

# Compute nodes
node=gpu
gname=gpu:v100:1

# Dirs
mkdir -p error
mkdir -p out
mkdir -p ${name}/error

echo "TEST"

# RUN these commands one by one and chnange job name accordignly (run A),B) first on continuous models)
jobname=${core}

# A) Produce predictions by splitting to multiple files
TEST=$(sbatch -p ${node} --job-name=split_${jobname} --gres ${gname} $traindir/predictions.sh ${sets}\
                        ${embeddingpath}/${emset}_${emdim}_${emkwd}${T} ${save} ${emdim} ${name} ${pos}\
                        ${traindir} | cut -f 4 -d' ')

## B) Bin terms by frequency
#FIRST=$(sbatch --job-name=terms $traindir/terms.sh ${name} ${sets}/set_0/ ${setdir} ${kwd} ${traindir} | cut -f 4 -d' ')

## C) Process predicitons
#SECOND=$(sbatch --mem 100000 --array=0-$(($(ls ${name}/preds/ | wc -l)/3-1))\
#               --job-name=results_${jobname} $traindir/results.sh ${sets} ${embeddingpath}/${emset}_${emdim}_${emkwd}${T}\
#               ${save} ${emdim} ${name} ${pos} ${traindir} | cut -f 4 -d' ')

## D) Aggregate processing
#if [ ${core} == traditional ]
#then
#  THIRD=$(sbatch --job-name=aggregate_${jobname} $traindir/bl_aggregate.sh\
#                  ${name} ${setdir} ${kwd} ${traindir} | cut -f 4 -d' ')
#else
#  THIRD=$(sbatch --job-name=aggregate_${jobname} $traindir/aggregate.sh\
#                  ${name} ${setdir} ${pos} ${kwd} ${traindir} | cut -f 4 -d' ')
#fi
