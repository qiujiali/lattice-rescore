#!/bin/bash
set -e

cmd=queue.pl

nj=${1}
dataset=${2}
recog_set=${3}
ngram=${4}
rnnlm=${5}
isca_1=${6}
isca_2=${7}
spm=${8}
tag=${9}
gsf=${10}

# activate virtualenv
. ./path.sh

# split json file
if [ ! -d "../exp_final/${dataset}/json/${recog_set}/split${nj}utt" ]; then
    splitjson.py \
        --parts ${nj} \
        ../exp_final/${dataset}/json/${recog_set}/data.json
fi

# submit jobs
latdir="../exp_final/${dataset}/lattice/${recog_set}"
latdir_out=${latdir}_latexp${ngram}g${tag}
mkdir -p ${latdir_out}/log
echo $0 $@ >> ${latdir_out}/log/CMD
${cmd} JOB=1:${nj} ${latdir_out}/log/get_lattice.JOB.log \
    ./get_lattice.py \
    ${latdir} \
    ${latdir_out} \
    ${ngram} \
    --rnnlm_path ${rnnlm} \
    --isca_path ${isca_1} \
    --isca_path ${isca_2} \
    --spm_path ${spm} \
    --js_path ../exp_final/${dataset}/json/${recog_set}/split${nj}utt/data.JOB.json \
    --gsf ${gsf} \
    --overwrite
    # --acronyms utils/acronyms.map \
