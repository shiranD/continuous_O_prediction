#!/bin/bash
set -x
set -e

main=../../.. # path to Shiran's main dir /home/groups/BedrickLab/shiran/gans/ 

# Data
sett=${main}/data # path to main data folder
emkwd=pos_nyt_big
kwd=pos_nyt_big # subdir in main data folder 
sets=${sett}/${kwd} # the full datapath

# Tensorboard dirname
tbrd=${kwd}

# SRC code
traindir=src #path to source code for training

# Embedding
embeddingpath=${main}/embeddings # path to embedding dir
#T=_T # (values: _t or nothing) if trans happens add it to the name
emset=w2v # type/name of embedding file
emdim=50 # embedding dimension

# Model architecture
typpe=LSTM # core RNN layer
nlayer=2 # number of layers
loss=CE # type of loss function (cos, mse, mm). irrelevant for traditional training (variable is ignored)
pos=False # (False/True values) apply POS decoding, irrelevant for traditional training (variable is ignored)

# Model Core approach
#core=continuous
#core=gan
core=traditional
#core=vae
#core=gan_vae

# Model name and path
modelpath=${kwd}_models # dirname of models (will be created)
name=${loss}_${typpe}${nlayer}${T}_${core}_${emdim}_${kwd}_${emset}_p # training name
save=$modelpath/${name} # modelname

# resume an experiment
more=False # assuming the model exist True resumes an experiment (False would start a new one)
#orig=orig_model/cos_LSTM2_continuous_50_pos_nyt_w2v
nameP=posnyt
# Compute nodes
node=gpu
gname=gpu:v100:1

# Dirs
mkdir -p ${modelpath}
mkdir -p reports
mkdir -p error
mkdir -p out

echo "TRAIN"
# Train a model
folds=0

echo "Training ${name}"

if [ "${core}" == "traditional" ]
then
  TRAIN=$(sbatch --array=0-$folds -p ${node} --job-name=${nameP} --gres ${gname} $traindir/${core}.sh\
        ${sets} ${embeddingpath}/${emset}_${emdim}_${emkwd}${T}_p ${save} ${emdim} ${name} ${typpe} ${nlayer}\
        ${tbrd} ${traindir} | cut -f 4 -d' ')
else
  TRAIN=$(sbatch --array=0-$folds -p ${node} --job-name=${nameP} --gres ${gname} ${traindir}/${core}.sh\
        ${sets} ${embeddingpath}/${emset}_${emdim}_${emkwd}${T} ${save} ${emdim} ${name} ${typpe} ${nlayer}\
        ${tbrd} ${pos} ${loss} ${more} ${traindir} ${orig} | cut -f 4 -d' ')
fi
